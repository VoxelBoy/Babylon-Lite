/**
 * Module: compute-shader
 *
 * A user-facing compute seam. Lite already runs compute internally (mipmap
 * generation, BRDF decode, thin-instance culling, HDR/IBL), but never exposed a
 * way for callers to run their own, so GPU-generated data, and GPU-generated
 * geometry in particular, had no supported path.
 *
 * Shape follows `createShaderMaterial`: the caller supplies WGSL plus declared
 * uniforms and storage bindings, and the object owns its bind-group layout,
 * pipeline, and uniform buffer. Nothing here leaks a raw WebGPU handle, since
 * storage is bound as an opaque `StorageBuffer`, so the "no GPU internals in the
 * public API" pillar holds.
 *
 * Zero cost when unused: no core render path imports this module.
 *
 * Uniforms are for values constant across a dispatch. PER-ITEM parameters belong
 * in a read-only storage buffer that the shader indexes by invocation id, which
 * is how ONE dispatch covers many items:
 *
 *     struct ChunkParams { slotBase: u32, u0: f32, v0: f32, span: f32 };
 *     @group(0) @binding(1) var<storage, read> params: array<ChunkParams>;
 *     let chunk = gid.x / vertsPerChunk;
 *     let p = params[chunk];
 *
 * That scales past a uniform buffer's binding-size limit, keeps GPU occupancy
 * high, and removes any need for per-dispatch uniform juggling. An earlier
 * revision let callers re-set uniforms between batched dispatches instead; it
 * silently produced wrong results, because `queue.writeBuffer` is ordered
 * against submission rather than recording, so every dispatch in a batch
 * observed the last values written.
 *
 * Ordering: WebGPU queue submission is ordered, so a dispatch submitted before a
 * frame's draws is visible to those draws.
 */
import type { EngineContext } from "../engine/engine.js";
import { computeUboLayout } from "../shader/ubo-layout.js";
import type { UboSpec } from "../shader/fragment-types.js";
import { createEmptyUniformBuffer } from "../resource/gpu-buffers.js";
import { _getStorageBufferHandle, type StorageBuffer } from "../resource/storage-buffer.js";
import type { ShaderDefineMap, ShaderUniformType, ShaderUniformValue } from "../material/shader/shader-material.js";

/** A uniform constant across the dispatch. Per-item data belongs in a storage buffer. */
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
    /** WGSL body. Uniform and storage declarations are generated and prepended. */
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
    /** @internal */ _uboData: ArrayBuffer | null;
    /** @internal */ _uboBuffer: GPUBuffer | null;
    /** @internal */ _uboDirty: boolean;
    /** @internal */ _bindings: Map<string, StorageBuffer>;
    /** @internal */ _layout: GPUBindGroupLayout | null;
    /** @internal */ _pipeline: GPUComputePipeline | null;
    /** @internal */ _bindGroup: GPUBindGroup | null;
    /** @internal */ _bindGroupDirty: boolean;
    /** @internal */ _destroyed: boolean;
}

function assertIdentifier(kind: string, name: string): void {
    if (!/^[A-Za-z_]\w*$/.test(name)) {
        throw new Error(`ComputeShader: ${kind} name "${name}" is not a valid WGSL identifier.`);
    }
}

function formatF32(value: number): string {
    return Number.isInteger(value) ? `${value}.0` : String(value);
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

/** Create a compute program. WGSL declarations for uniforms and storage are generated. */
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
        _uboBuffer: uboSpec ? createEmptyUniformBuffer(engine, uboSpec._totalBytes, `${options.name ?? "compute"}-ubo`) : null,
        _uboDirty: true,
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
        entries.push({ binding: binding++, visibility: COMPUTE, buffer: { type: "uniform" } });
    }
    for (const decl of shader._storageDecls) {
        entries.push({ binding: binding++, visibility: COMPUTE, buffer: { type: decl.writable ? "storage" : "read-only-storage" } });
    }
    shader._layout = shader._engine._device.createBindGroupLayout({ label: `${shader.name}-layout`, entries });
    return shader._layout;
}

function pipelineDescriptor(shader: ComputeShader): GPUComputePipelineDescriptor {
    const device = shader._engine._device;
    const module = device.createShaderModule({ code: shader._wgsl, label: `${shader.name}-module` });
    return {
        label: shader.name,
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout(shader)] }),
        compute: { module, entryPoint: shader._entryPoint },
    };
}

function ensurePipeline(shader: ComputeShader): GPUComputePipeline {
    shader._pipeline ??= shader._engine._device.createComputePipeline(pipelineDescriptor(shader));
    return shader._pipeline;
}

/**
 * Compile the pipeline off the critical path.
 *
 * `dispatchCompute` compiles synchronously on first use, which stalls the frame
 * it happens on. Awaiting this beforehand moves that cost elsewhere.
 */
export async function prepareComputeShader(shader: ComputeShader): Promise<void> {
    if (shader._pipeline || shader._destroyed) return;
    shader._pipeline = await shader._engine._device.createComputePipelineAsync(pipelineDescriptor(shader));
}

/** Set a declared uniform. Uniforms are constant across a dispatch. */
export function setComputeUniform(shader: ComputeShader, name: string, value: ShaderUniformValue): void {
    const spec = shader._uboSpec;
    if (!spec || !shader._uboData) throw new Error(`ComputeShader "${shader.name}": no uniforms were declared.`);
    const offset = spec._offsets.get(name);
    if (offset === undefined) throw new Error(`ComputeShader "${shader.name}": uniform "${name}" was not declared.`);
    const decl = shader._uniformDecls.find((u) => u.name === name)!;
    const view = new DataView(shader._uboData);
    const nums = typeof value === "number" ? [value] : Array.from(value as ArrayLike<number>);
    for (let i = 0; i < nums.length; i++) {
        const at = offset + i * 4;
        if (at + 4 > shader._uboData.byteLength) break;
        if (decl.type === "u32") view.setUint32(at, nums[i]!, true);
        else if (decl.type === "i32") view.setInt32(at, nums[i]!, true);
        else view.setFloat32(at, nums[i]!, true);
    }
    shader._uboDirty = true;
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
    const entries: GPUBindGroupEntry[] = [];
    let binding = 0;
    if (shader._uboBuffer) entries.push({ binding: binding++, resource: { buffer: shader._uboBuffer } });
    for (const decl of shader._storageDecls) {
        const bound = shader._bindings.get(decl.name);
        if (!bound) throw new Error(`ComputeShader "${shader.name}": storage buffer "${decl.name}" was declared but never bound.`);
        entries.push({ binding: binding++, resource: { buffer: _getStorageBufferHandle(shader._engine, bound) } });
    }
    shader._bindGroup = shader._engine._device.createBindGroup({ label: `${shader.name}-bindgroup`, layout: bindGroupLayout(shader), entries });
    shader._bindGroupDirty = false;
    return shader._bindGroup;
}

/**
 * Run the compute program over `x` by `y` by `z` workgroups and submit it.
 *
 * Cover many items in ONE dispatch by sizing the workgroup count to the whole
 * work set and indexing per-item parameters out of a storage buffer.
 */
export function dispatchCompute(engine: EngineContext, shader: ComputeShader, x: number, y = 1, z = 1): void {
    if (shader._destroyed) throw new Error(`ComputeShader "${shader.name}" has been disposed.`);
    if (shader._engine !== engine) throw new Error(`ComputeShader "${shader.name}" belongs to a different engine.`);
    if (!(x > 0 && y > 0 && z > 0)) throw new Error(`ComputeShader "${shader.name}": workgroup counts must all be positive.`);

    const device = engine._device;
    if (shader._uboDirty && shader._uboBuffer && shader._uboData) {
        device.queue.writeBuffer(shader._uboBuffer, 0, shader._uboData);
        shader._uboDirty = false;
    }

    const pipeline = ensurePipeline(shader);
    const bindGroup = ensureBindGroup(shader);

    const encoder = device.createCommandEncoder({ label: `${shader.name}-encoder` });
    const pass = encoder.beginComputePass({ label: `${shader.name}-pass` });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
    device.queue.submit([encoder.finish()]);
}

/** Release the program's GPU objects. Bound storage buffers are not owned or freed. */
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

// ─── Many binding sets per program, one submission per batch ────────────────
//
// A `ComputeShader` carries exactly one set of bound buffers, so running the
// same program over several independent data sets meant creating several
// programs — and a separate pipeline for each, for identical WGSL. And every
// `dispatchCompute` builds its own encoder and submits, so N data sets cost N
// submissions per frame where one would do.
//
// Both matter for the case this seam exists to serve. A GPU-generated world
// batches its work per source object (per body, per volume, per emitter): the
// program is one, the parameter buffers are many, and they are dispatched
// together. `ComputeBindings` separates "which program" from "which data", and
// `dispatchComputeBatch` records a whole frame's dispatches into one command
// buffer.

declare const computeBindingsBrand: unique symbol;

/** A reusable set of storage bindings for one `ComputeShader`. Create with {@link createComputeBindings}. */
export interface ComputeBindings {
    readonly [computeBindingsBrand]: true;
    readonly name: string;
    /** @internal */ readonly _shader: ComputeShader;
    /** @internal */ readonly _bindings: Map<string, StorageBuffer>;
    /** @internal */ _bindGroup: GPUBindGroup | null;
    /** @internal */ _dirty: boolean;
    /** @internal */ _destroyed: boolean;
}

/**
 * Create an additional binding set for `shader`.
 *
 * The program's pipeline, layout and uniform buffer are shared; only the bound
 * storage buffers differ. Uniforms stay on the shader, since they are constant
 * across a dispatch by definition — per-set data belongs in a storage buffer.
 */
export function createComputeBindings(shader: ComputeShader, name?: string): ComputeBindings {
    if (shader._destroyed) throw new Error(`ComputeShader "${shader.name}" has been disposed.`);
    return {
        name: name ?? `${shader.name}-bindings`,
        _shader: shader,
        _bindings: new Map<string, StorageBuffer>(),
        _bindGroup: null,
        _dirty: true,
        _destroyed: false,
    } as unknown as ComputeBindings;
}

/** Bind a storage allocation within a binding set. */
export function setComputeBindingsStorageBuffer(bindings: ComputeBindings, name: string, buffer: StorageBuffer): void {
    if (bindings._destroyed) throw new Error(`ComputeBindings "${bindings.name}" has been disposed.`);
    const shader = bindings._shader;
    const decl = shader._storageDecls.find((d) => d.name === name);
    if (!decl) throw new Error(`ComputeBindings "${bindings.name}": storage buffer "${name}" was not declared on "${shader.name}".`);
    if (decl.writable && !buffer._writable) {
        throw new Error(`ComputeBindings "${bindings.name}": binding "${name}" is read_write, so its StorageBuffer must be created with { writable: true }.`);
    }
    bindings._bindings.set(name, buffer);
    bindings._dirty = true;
}

/** Release the binding set. The program and the bound buffers are not owned or freed. */
export function disposeComputeBindings(bindings: ComputeBindings): void {
    bindings._bindGroup = null;
    bindings._bindings.clear();
    bindings._destroyed = true;
}

/** One dispatch within a {@link dispatchComputeBatch}. */
export interface ComputeDispatch {
    readonly shader: ComputeShader;
    /** Binding set to run against. Defaults to the shader's own bindings. */
    readonly bindings?: ComputeBindings;
    readonly x: number;
    readonly y?: number;
    readonly z?: number;
}

/**
 * Record several dispatches into ONE command buffer and submit once.
 *
 * Equivalent to calling `dispatchCompute` per item, minus the per-item encoder
 * and submission. Dispatches execute in array order.
 *
 * Uniform writes are flushed once per shader before recording begins. That is
 * the same ordering hazard `dispatchCompute` documents: `queue.writeBuffer` is
 * ordered against submission, not recording, so two dispatches in one batch
 * cannot see different uniform values — vary per-dispatch data through the
 * binding set's storage buffers instead.
 */
export function dispatchComputeBatch(engine: EngineContext, dispatches: readonly ComputeDispatch[]): void {
    if (dispatches.length === 0) return;
    const device = engine._device;
    const flushed = new Set<ComputeShader>();
    for (const d of dispatches) {
        const shader = d.shader;
        if (shader._destroyed) throw new Error(`ComputeShader "${shader.name}" has been disposed.`);
        if (shader._engine !== engine) throw new Error(`ComputeShader "${shader.name}" belongs to a different engine.`);
        if (!(d.x > 0 && (d.y ?? 1) > 0 && (d.z ?? 1) > 0)) throw new Error(`ComputeShader "${shader.name}": workgroup counts must all be positive.`);
        if (d.bindings && d.bindings._shader !== shader) {
            throw new Error(`ComputeBindings "${d.bindings.name}" belongs to a different ComputeShader than "${shader.name}".`);
        }
        if (!flushed.has(shader)) {
            flushed.add(shader);
            if (shader._uboDirty && shader._uboBuffer && shader._uboData) {
                device.queue.writeBuffer(shader._uboBuffer, 0, shader._uboData);
                shader._uboDirty = false;
            }
        }
    }

    const encoder = device.createCommandEncoder({ label: `${dispatches[0]!.shader.name}-batch-encoder` });
    const pass = encoder.beginComputePass({ label: `${dispatches[0]!.shader.name}-batch-pass` });
    let lastPipeline: GPUComputePipeline | null = null;
    for (const d of dispatches) {
        const pipeline = ensurePipeline(d.shader);
        if (pipeline !== lastPipeline) {
            pass.setPipeline(pipeline);
            lastPipeline = pipeline;
        }
        pass.setBindGroup(0, d.bindings ? ensureBindingsGroup(d.bindings) : ensureBindGroup(d.shader));
        pass.dispatchWorkgroups(d.x, d.y ?? 1, d.z ?? 1);
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
}

function ensureBindingsGroup(bindings: ComputeBindings): GPUBindGroup {
    if (bindings._bindGroup && !bindings._dirty) return bindings._bindGroup;
    if (bindings._destroyed) throw new Error(`ComputeBindings "${bindings.name}" has been disposed.`);
    const shader = bindings._shader;
    const entries: GPUBindGroupEntry[] = [];
    let binding = 0;
    if (shader._uboBuffer) entries.push({ binding: binding++, resource: { buffer: shader._uboBuffer } });
    for (const decl of shader._storageDecls) {
        const bound = bindings._bindings.get(decl.name);
        if (!bound) throw new Error(`ComputeBindings "${bindings.name}": storage buffer "${decl.name}" was declared but never bound.`);
        entries.push({ binding: binding++, resource: { buffer: _getStorageBufferHandle(shader._engine, bound) } });
    }
    bindings._bindGroup = shader._engine._device.createBindGroup({ label: `${bindings.name}-bindgroup`, layout: bindGroupLayout(shader), entries });
    bindings._dirty = false;
    return bindings._bindGroup;
}
