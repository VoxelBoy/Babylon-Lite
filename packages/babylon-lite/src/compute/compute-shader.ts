/**
 * Module: compute-shader
 *
 * A user-facing compute seam. Lite already runs compute internally (mipmap
 * generation, BRDF decode, thin-instance culling, HDR/IBL), but never exposed a
 * way for callers to run their own — so GPU-generated data, and in particular
 * GPU-generated geometry, had no supported path.
 *
 * Shape follows `createShaderMaterial`: the caller supplies WGSL plus declared
 * uniforms and storage buffers, and the object owns its bind-group layout,
 * pipeline, and uniform buffer. Nothing here leaks a raw WebGPU handle — storage
 * is bound as an opaque `StorageBuffer`, satisfying the "no GPU internals in the
 * public API" pillar.
 *
 * Zero cost when unused: no core render path imports this module.
 *
 * Ordering: WebGPU queue submission is ordered, so a dispatch submitted before a
 * frame's draws is visible to those draws. `dispatchCompute` submits immediately
 * by default — the safe behaviour. Wrap several dispatches in
 * `beginComputeBatch`/`endComputeBatch` to record them into ONE encoder and pay a
 * single submit, which is what a chunked terrain filler wants.
 */
import type { EngineContext } from "../engine/engine.js";
import { computeUboLayout } from "../shader/ubo-layout.js";
import type { UboSpec } from "../shader/fragment-types.js";
import { _getStorageBufferHandle, type StorageBuffer } from "../resource/storage-buffer.js";
import type { ShaderDefineMap, ShaderUniformType, ShaderUniformValue } from "../material/shader/shader-material.js";

/** A uniform available to the compute entry point. */
export interface ComputeUniformDecl {
    readonly name: string;
    readonly type: ShaderUniformType;
    readonly defaultValue?: ShaderUniformValue;
}

/** A storage binding. `type` is the WGSL variable type, e.g. `array<vec4<f32>>`. */
export interface ComputeStorageBufferDecl {
    readonly name: string;
    readonly type: string;
    /** Bind as `var<storage, read_write>`. The bound allocation must be `writable: true`. */
    readonly writable?: boolean;
}

/** Options for {@link createComputeShader}. */
export interface ComputeShaderOptions {
    readonly name?: string;
    /** WGSL body. The uniform/storage declarations are generated and prepended. */
    readonly computeSource: string;
    /** Entry point name. Default `"main"`. */
    readonly entryPoint?: string;
    readonly uniforms?: readonly ComputeUniformDecl[];
    readonly storageBuffers?: readonly ComputeStorageBufferDecl[];
    readonly defines?: ShaderDefineMap;
}

declare const computeShaderBrand: unique symbol;

/** A compute program with its own bindings. Create with {@link createComputeShader}. */
export interface ComputeShader {
    readonly [computeShaderBrand]: true;
    readonly name: string;
    /** @internal */ readonly _engine: EngineContext;
    /** @internal */ readonly _entryPoint: string;
    /** @internal */ readonly _wgsl: string;
    /** @internal */ readonly _uniformDecls: readonly ComputeUniformDecl[];
    /** @internal */ readonly _storageDecls: readonly ComputeStorageBufferDecl[];
    /** @internal */ readonly _uboSpec: UboSpec | null;
    /** @internal Staging copy of the current uniform values. */
    _uboData: ArrayBuffer | null;
    /** @internal Ring of per-dispatch uniform slots (see the dynamic-offset note). */
    _uboBuffer: GPUBuffer | null;
    /** @internal Aligned byte stride between ring slots. */
    _uboStride: number;
    /** @internal Ring capacity in slots. */
    _uboSlots: number;
    /** @internal Next free ring slot for this submit. */
    _uboCursor: number;
    /** @internal */ _bindings: Map<string, StorageBuffer>;
    /** @internal */ _layout: GPUBindGroupLayout | null;
    /** @internal */ _pipeline: GPUComputePipeline | null;
    /** @internal */ _bindGroup: GPUBindGroup | null;
    /** @internal */ _bindGroupDirty: boolean;
    /** @internal */ _destroyed: boolean;
}

/** @internal Batch state, kept on the engine so unused batching costs nothing. */
interface ComputeBatchState {
    encoder: GPUCommandEncoder | null;
    depth: number;
    /** Shaders that took uniform-ring slots in this batch; reset once it submits. */
    shaders?: Set<ComputeShader>;
}
const batches = new WeakMap<object, ComputeBatchState>();

function assertIdentifier(kind: string, name: string): void {
    if (!/^[A-Za-z_]\w*$/.test(name)) {
        throw new Error(`ComputeShader: ${kind} name "${name}" is not a valid WGSL identifier.`);
    }
}

function buildPrelude(options: ComputeShaderOptions, uboSpec: UboSpec | null): string {
    let wgsl = "";
    for (const [name, value] of Object.entries(options.defines ?? {})) {
        assertIdentifier("define", name);
        wgsl += `const ${name}: ${typeof value === "boolean" ? "bool" : "f32"} = ${typeof value === "boolean" ? String(value) : formatF32(value)};\n`;
    }
    let binding = 0;
    if (uboSpec) {
        wgsl += `struct ComputeUniforms {\n${uboSpec._structBody}\n}\n@group(0) @binding(${binding++}) var<uniform> uniforms: ComputeUniforms;\n`;
    }
    for (const storage of options.storageBuffers ?? []) {
        assertIdentifier("storage buffer", storage.name);
        wgsl += `@group(0) @binding(${binding++}) var<storage, ${storage.writable ? "read_write" : "read"}> ${storage.name}: ${storage.type};\n`;
    }
    return wgsl;
}

function formatF32(value: number): string {
    return Number.isInteger(value) ? `${value}.0` : String(value);
}

function alignTo(n: number, to: number): number {
    return Math.ceil(n / to) * to;
}

function uniformOffsetAlignment(engine: EngineContext): number {
    return engine._device.limits?.minUniformBufferOffsetAlignment ?? 256;
}

/**
 * Grow the per-dispatch uniform ring to hold at least `slots` entries.
 *
 * Every dispatch needs its OWN copy of the uniform values. `queue.writeBuffer`
 * is ordered against submission, not against recording, so N dispatches sharing
 * one uniform buffer all observe the LAST values written — which silently
 * collapses a batch of differently-parameterised dispatches into N copies of the
 * final one. Each dispatch therefore takes a slot in this ring and binds it with
 * a dynamic offset.
 */
function ensureUboCapacity(shader: ComputeShader, slots: number): void {
    if (!shader._uboSpec || slots <= shader._uboSlots) return;
    const engine = shader._engine;
    const next = Math.max(slots, shader._uboSlots * 2, 16);
    shader._uboBuffer?.destroy();
    shader._uboBuffer = engine._device.createBuffer({
        label: `${shader.name}-ubo-ring`,
        size: shader._uboStride * next,
        usage: globalThis.GPUBufferUsage.UNIFORM | globalThis.GPUBufferUsage.COPY_DST,
    });
    shader._uboSlots = next;
    shader._bindGroupDirty = true; // the bind group references the old buffer
}

/** Create a compute program. The WGSL declarations for uniforms/storage are generated. */
export function createComputeShader(engine: EngineContext, options: ComputeShaderOptions): ComputeShader {
    const uniformDecls = options.uniforms ?? [];
    for (const u of uniformDecls) {
        assertIdentifier("uniform", u.name);
    }
    const uboSpec = uniformDecls.length > 0 ? computeUboLayout(uniformDecls.map((u) => ({ _name: u.name, _type: u.type }))) : null;

    const shader = {
        name: options.name ?? "compute",
        _engine: engine,
        _entryPoint: options.entryPoint ?? "main",
        _wgsl: buildPrelude(options, uboSpec) + options.computeSource,
        _uniformDecls: uniformDecls,
        _storageDecls: options.storageBuffers ?? [],
        _uboSpec: uboSpec,
        _uboData: uboSpec ? new ArrayBuffer(uboSpec._totalBytes) : null,
        _uboBuffer: null,
        _uboStride: uboSpec ? alignTo(uboSpec._totalBytes, uniformOffsetAlignment(engine)) : 0,
        _uboSlots: 0,
        _uboCursor: 0,
        _bindings: new Map<string, StorageBuffer>(),
        _layout: null,
        _pipeline: null,
        _bindGroup: null,
        _bindGroupDirty: true,
        _destroyed: false,
    } as unknown as ComputeShader;

    for (const u of uniformDecls) {
        if (u.defaultValue !== undefined) {
            setComputeUniform(shader, u.name, u.defaultValue);
        }
    }
    return shader;
}

function bindGroupLayout(shader: ComputeShader): GPUBindGroupLayout {
    if (shader._layout) return shader._layout;
    const entries: GPUBindGroupLayoutEntry[] = [];
    let binding = 0;
    const COMPUTE = globalThis.GPUShaderStage.COMPUTE;
    if (shader._uboSpec) {
        // Dynamic offset: each dispatch binds its own slot of the uniform ring.
        entries.push({ binding: binding++, visibility: COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: shader._uboSpec._totalBytes } });
    }
    for (const decl of shader._storageDecls) {
        entries.push({ binding: binding++, visibility: COMPUTE, buffer: { type: decl.writable ? "storage" : "read-only-storage" } });
    }
    shader._layout = shader._engine._device.createBindGroupLayout({ label: `${shader.name}-layout`, entries });
    return shader._layout;
}

function ensurePipeline(shader: ComputeShader): GPUComputePipeline {
    if (shader._pipeline) return shader._pipeline;
    const device = shader._engine._device;
    const module = device.createShaderModule({ code: shader._wgsl, label: `${shader.name}-module` });
    shader._pipeline = device.createComputePipeline({
        label: shader.name,
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout(shader)] }),
        compute: { module, entryPoint: shader._entryPoint },
    });
    return shader._pipeline;
}

/**
 * Compile the pipeline off the critical path.
 *
 * `dispatchCompute` compiles synchronously on first use, which stalls the frame
 * it happens on. Awaiting this beforehand moves that cost off the hot path.
 */
export async function prepareComputeShader(shader: ComputeShader): Promise<void> {
    if (shader._pipeline || shader._destroyed) return;
    const device = shader._engine._device;
    const module = device.createShaderModule({ code: shader._wgsl, label: `${shader.name}-module` });
    shader._pipeline = await device.createComputePipelineAsync({
        label: shader.name,
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout(shader)] }),
        compute: { module, entryPoint: shader._entryPoint },
    });
}

/** Set a declared uniform value. */
export function setComputeUniform(shader: ComputeShader, name: string, value: ShaderUniformValue): void {
    const spec = shader._uboSpec;
    if (!spec || !shader._uboData) throw new Error(`ComputeShader "${shader.name}": no uniforms were declared.`);
    const offset = spec._offsets.get(name);
    if (offset === undefined) throw new Error(`ComputeShader "${shader.name}": uniform "${name}" was not declared.`);
    const decl = shader._uniformDecls.find((u) => u.name === name)!;
    const view = new DataView(shader._uboData);
    const nums = typeof value === "number" ? [value] : Array.from(value as ArrayLike<number>);
    const int = decl.type === "u32" || decl.type === "i32";
    for (let i = 0; i < nums.length; i++) {
        const at = offset + i * 4;
        if (at + 4 > shader._uboData.byteLength) break;
        if (int) {
            if (decl.type === "u32") view.setUint32(at, nums[i]!, true);
            else view.setInt32(at, nums[i]!, true);
        } else {
            view.setFloat32(at, nums[i]!, true);
        }
    }
    // Values are snapshotted into a ring slot at dispatch time, so no dirty flag.
}

/** Bind a storage allocation to a declared storage binding. */
export function setComputeStorageBuffer(shader: ComputeShader, name: string, buffer: StorageBuffer): void {
    const decl = shader._storageDecls.find((d) => d.name === name);
    if (!decl) throw new Error(`ComputeShader "${shader.name}": storage buffer "${name}" was not declared.`);
    if (decl.writable && !buffer._writable) {
        throw new Error(`ComputeShader "${shader.name}": binding "${name}" is read_write, so its StorageBuffer must be created with { writable: true }.`);
    }
    shader._bindings.set(name, buffer);
    shader._bindGroupDirty = true;
}

function ensureBindGroup(shader: ComputeShader): GPUBindGroup {
    if (shader._bindGroup && !shader._bindGroupDirty) return shader._bindGroup;
    const device = shader._engine._device;
    const entries: GPUBindGroupEntry[] = [];
    let binding = 0;
    if (shader._uboBuffer) entries.push({ binding: binding++, resource: { buffer: shader._uboBuffer, offset: 0, size: shader._uboSpec!._totalBytes } });
    for (const decl of shader._storageDecls) {
        const bound = shader._bindings.get(decl.name);
        if (!bound) throw new Error(`ComputeShader "${shader.name}": storage buffer "${decl.name}" was declared but never bound.`);
        entries.push({ binding: binding++, resource: { buffer: _getStorageBufferHandle(shader._engine, bound) } });
    }
    shader._bindGroup = device.createBindGroup({ label: `${shader.name}-bindgroup`, layout: bindGroupLayout(shader), entries });
    shader._bindGroupDirty = false;
    return shader._bindGroup;
}

/** Record several dispatches into one encoder; pair with {@link endComputeBatch}. */
export function beginComputeBatch(engine: EngineContext): void {
    const state = batches.get(engine) ?? { encoder: null, depth: 0 };
    if (state.depth === 0) {
        state.encoder = engine._device.createCommandEncoder({ label: "compute-batch" });
    }
    state.depth++;
    batches.set(engine, state);
}

/** Submit the batch opened by {@link beginComputeBatch}. */
export function endComputeBatch(engine: EngineContext): void {
    const state = batches.get(engine);
    if (!state || state.depth === 0) throw new Error("endComputeBatch called without a matching beginComputeBatch.");
    state.depth--;
    if (state.depth === 0 && state.encoder) {
        engine._device.queue.submit([state.encoder.finish()]);
        state.encoder = null;
        // The batch is submitted, so every uniform slot it consumed is free again.
        for (const shader of state.shaders ?? []) shader._uboCursor = 0;
        state.shaders?.clear();
    }
}

/**
 * Run the compute program over `x`×`y`×`z` workgroups.
 *
 * Submits immediately unless a batch is open. Queue submission is ordered, so a
 * dispatch issued before a frame's draws is visible to those draws.
 */
export function dispatchCompute(engine: EngineContext, shader: ComputeShader, x: number, y = 1, z = 1): void {
    if (shader._destroyed) throw new Error(`ComputeShader "${shader.name}" has been disposed.`);
    if (shader._engine !== engine) throw new Error(`ComputeShader "${shader.name}" belongs to a different engine.`);
    if (!(x > 0 && y > 0 && z > 0)) throw new Error(`ComputeShader "${shader.name}": workgroup counts must all be positive.`);

    const device = engine._device;
    const batch = batches.get(engine);
    const batching = !!batch && batch.depth > 0 && !!batch.encoder;

    // Snapshot this dispatch's uniform values into their own ring slot.
    let dynamicOffset = 0;
    if (shader._uboSpec && shader._uboData) {
        ensureUboCapacity(shader, shader._uboCursor + 1);
        dynamicOffset = shader._uboCursor * shader._uboStride;
        device.queue.writeBuffer(shader._uboBuffer!, dynamicOffset, shader._uboData);
        shader._uboCursor++;
    }

    const pipeline = ensurePipeline(shader);
    const bindGroup = ensureBindGroup(shader);
    const encoder = batching ? batch!.encoder! : device.createCommandEncoder({ label: `${shader.name}-encoder` });

    const pass = encoder.beginComputePass({ label: `${shader.name}-pass` });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup, shader._uboSpec ? [dynamicOffset] : []);
    pass.dispatchWorkgroups(x, y, z);
    pass.end();

    if (!batching) {
        device.queue.submit([encoder.finish()]);
        // Slots are only safe to reuse once the work referencing them is submitted.
        shader._uboCursor = 0;
    } else {
        (batch!.shaders ??= new Set()).add(shader);
    }
}

/** Release the program's GPU objects. Bound storage buffers are NOT owned or freed. */
export function disposeComputeShader(shader: ComputeShader): void {
    if (shader._destroyed) return;
    shader._uboBuffer?.destroy();
    shader._uboBuffer = null;
    shader._uboData = null;
    shader._pipeline = null;
    shader._bindGroup = null;
    shader._layout = null;
    shader._bindings.clear();
    shader._destroyed = true;
}
