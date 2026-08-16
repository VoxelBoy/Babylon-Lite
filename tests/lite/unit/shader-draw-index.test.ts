import { describe, expect, it, vi } from "vitest";

import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import type { RenderTargetSignature } from "../../../packages/babylon-lite/src/engine/render-target";
import { createShaderMaterial } from "../../../packages/babylon-lite/src/material/shader/shader-material";
import { getOrCreateShaderPipeline, getOrCreateShaderPipelineBindings } from "../../../packages/babylon-lite/src/material/shader/shader-pipeline";
import { clearShaderPipelineCache } from "../../../packages/babylon-lite/src/material/shader/shader-pipeline-cache";
import { buildShaderMaterialRenderables } from "../../../packages/babylon-lite/src/material/shader/shader-renderable";
import { initMeshTransform } from "../../../packages/babylon-lite/src/mesh/mesh";
import type { SceneContext } from "../../../packages/babylon-lite/src/scene/scene-core";

const VERTEX = `@vertex fn mainVertex(input: VertexInput) -> @builtin(position) vec4<f32> {
  return shaderSystem.worldViewProjection * vec4<f32>(input.position, 1.0);
}`;
const FRAGMENT = `@fragment fn mainFragment() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }`;

function makeEngine() {
    const createShaderModule = vi.fn((descriptor: GPUShaderModuleDescriptor) => descriptor as unknown as GPUShaderModule);
    const device = {
        createBindGroupLayout: vi.fn((d: GPUBindGroupLayoutDescriptor) => d as unknown as GPUBindGroupLayout),
        createPipelineLayout: vi.fn((d: GPUPipelineLayoutDescriptor) => d as unknown as GPUPipelineLayout),
        createShaderModule,
        createRenderPipeline: vi.fn((d: GPURenderPipelineDescriptor) => d as unknown as GPURenderPipeline),
        createBindGroup: vi.fn((d: GPUBindGroupDescriptor) => d as unknown as GPUBindGroup),
        createBuffer: vi.fn((d: GPUBufferDescriptor) => ({ size: Number(d.size), destroy: vi.fn() }) as unknown as GPUBuffer),
        queue: { writeBuffer: vi.fn() },
    } as unknown as GPUDevice;
    const engine = { _device: device, _disposables: [] } as unknown as EngineContext;
    Object.assign(engine, { engine });
    return { engine, createShaderModule };
}

const TARGET: RenderTargetSignature = { _colorFormat: "bgra8unorm", _depthStencilFormat: "depth24plus", _sampleCount: 1 };

function generatedWgsl(drawIndex: boolean): string {
    clearShaderPipelineCache();
    const { engine, createShaderModule } = makeEngine();
    const material = createShaderMaterial({
        name: `drawIndex-${drawIndex}`,
        vertexSource: VERTEX,
        fragmentSource: FRAGMENT,
        attributes: ["position"],
        ...(drawIndex ? { drawIndex: true } : {}),
    });
    getOrCreateShaderPipeline(engine, TARGET, material, getOrCreateShaderPipelineBindings(engine, material));
    return createShaderModule.mock.calls.map((c) => (c[0] as GPUShaderModuleDescriptor).code).join("\n");
}

describe("ShaderMaterialOptions.drawIndex", () => {
    it("emits the instance_index builtin in VertexInput when requested", () => {
        expect(generatedWgsl(true)).toContain("@builtin(instance_index) drawIndex: u32,");
    });

    it("emits nothing when not requested, so existing shaders are byte-identical", () => {
        const without = generatedWgsl(false);
        expect(without).not.toContain("instance_index");
        // And the builtin is the ONLY difference — a shader that opts in must not
        // pick up any other change to its generated prelude.
        // replaceAll, not replace: the prelude is emitted into both the vertex
        // and the fragment module, so the line appears twice.
        expect(generatedWgsl(true).replaceAll("@builtin(instance_index) drawIndex: u32,\n", "")).toBe(without);
    });

    it("declares the builtin as a builtin, not as a vertex buffer slot", () => {
        clearShaderPipelineCache();
        const { engine } = makeEngine();
        const material = createShaderMaterial({
            name: "slots",
            vertexSource: VERTEX,
            fragmentSource: FRAGMENT,
            attributes: ["position"],
            drawIndex: true,
        });
        const pipeline = getOrCreateShaderPipeline(engine, TARGET, material, getOrCreateShaderPipelineBindings(engine, material)) as unknown as GPURenderPipelineDescriptor;
        // One attribute in, one vertex buffer out. The builtin costs no slot.
        expect(pipeline.vertex.buffers).toHaveLength(1);
    });
});

// The generated WGSL above proves the shader can READ a per-draw index. These
// prove the draw call actually SUPPLIES one — the half that a shader-source
// check cannot see, and the half that decides whether every mesh silently reads
// row 0.
function drawFixture(drawIndices: (number | undefined)[]): { drawIndexed: ReturnType<typeof vi.fn>; draw: () => void } {
    const drawIndexed = vi.fn();
    const { engine } = makeEngine();
    Object.assign(engine, { canvas: { width: 64, height: 64 } });
    const material = createShaderMaterial({
        vertexSource: VERTEX,
        fragmentSource: FRAGMENT,
        attributes: ["position"],
        drawIndex: true,
    });
    const meshes = drawIndices.map((drawIndex, i) =>
        initMeshTransform({
            name: `chunk${i}`,
            children: [],
            material,
            receiveShadows: false,
            ...(drawIndex === undefined ? {} : { drawIndex }),
            _gpu: {
                positionBuffer: {} as GPUBuffer,
                normalBuffer: {} as GPUBuffer,
                uvBuffer: {} as GPUBuffer,
                indexBuffer: {} as GPUBuffer,
                indexCount: 3,
                indexFormat: "uint32",
            },
        })
    );
    const scene = {
        surface: { engine },
        camera: null,
        _meshDisposables: new Map(),
        _meshAuxDisposables: new Map(),
    } as unknown as SceneContext;
    const result = buildShaderMaterialRenderables(scene, meshes);
    const binding = result.renderables[0]!.bind(engine, { _colorFormat: "rgba8unorm", _sampleCount: 1 } as RenderTargetSignature);
    const pass = { setVertexBuffer: vi.fn(), setIndexBuffer: vi.fn(), setBindGroup: vi.fn(), drawIndexed } as unknown as GPURenderPassEncoder;
    return { drawIndexed, draw: () => void binding.draw(pass, engine) };
}

describe("Mesh.drawIndex", () => {
    it("reaches the draw call as firstInstance, one row per mesh", () => {
        const { drawIndexed, draw } = drawFixture([0, 1, 2, 3]);
        draw();
        // drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance)
        expect(drawIndexed.mock.calls.map((c) => c[4])).toEqual([undefined, 1, 2, 3]);
        // Index 0 is the canonical case and must stay on the short call, so
        // meshes that never set drawIndex keep a byte-identical hot path.
        expect(drawIndexed.mock.calls[0]).toEqual([3]);
    });

    it("is mutable, so a mesh can be reassigned to another row without a rebuild", () => {
        const { drawIndexed, draw } = drawFixture([7]);
        draw();
        expect(drawIndexed.mock.calls[0]![4]).toBe(7);
    });

    it("leaves meshes that never set it on the canonical draw path", () => {
        const { drawIndexed, draw } = drawFixture([undefined, undefined]);
        draw();
        expect(drawIndexed.mock.calls).toEqual([[3], [3]]);
    });
});
