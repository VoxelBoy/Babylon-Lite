import { describe, expect, it, vi } from "vitest";

import type { EngineContext, SurfaceContext } from "../../../packages/babylon-lite/src/engine/engine";
import { createSceneContext } from "../../../packages/babylon-lite/src/scene/scene-core";

/** Capture the dFormat of every render target the default task allocates. */
function makeSurface(msaaSamples: number): { surface: SurfaceContext; formats: () => (string | undefined)[] } {
    const created: (string | undefined)[] = [];
    const engine = {
        _device: {
            createTexture: vi.fn(() => ({ createView: vi.fn(), destroy: vi.fn() }) as unknown as GPUTexture),
            createBindGroupLayout: vi.fn((d: GPUBindGroupLayoutDescriptor) => d as unknown as GPUBindGroupLayout),
            createPipelineLayout: vi.fn((d: GPUPipelineLayoutDescriptor) => d as unknown as GPUPipelineLayout),
            createBindGroup: vi.fn((d: GPUBindGroupDescriptor) => d as unknown as GPUBindGroup),
            createShaderModule: vi.fn((d: GPUShaderModuleDescriptor) => d as unknown as GPUShaderModule),
            createRenderPipeline: vi.fn((d: GPURenderPipelineDescriptor) => d as unknown as GPURenderPipeline),
            createSampler: vi.fn((d: GPUSamplerDescriptor) => d as unknown as GPUSampler),
            createBuffer: vi.fn((d: GPUBufferDescriptor) => ({ size: Number(d.size), destroy: vi.fn() }) as unknown as GPUBuffer),
            queue: { writeBuffer: vi.fn() },
            limits: {},
        } as unknown as GPUDevice,
    } as unknown as EngineContext;
    const surface = {
        engine,
        format: "bgra8unorm",
        msaaSamples,
        canvas: { width: 64, height: 64 },
        scRT: { _descriptor: { format: "bgra8unorm", samples: 1 } },
    } as unknown as SurfaceContext;
    Object.assign(engine, surface);
    // createRenderTarget is pure state, so the descriptors are readable without a GPU.
    return { surface, formats: () => created };
}

describe("SceneContextOptions.depthFormat", () => {
    // Assert the format the task reports for PIPELINE SIGNATURE MATCHING, not the
    // descriptor we passed in. That is the value the GPU actually sees, so it
    // catches the format being dropped anywhere between the option and the pass.
    const depthFormatsOf = (scene: unknown): string[] =>
        ((scene as { _frameGraph: { _tasks: { _targetSignature?: { _depthStencilFormat?: string } }[] } })._frameGraph._tasks || [])
            .map((t) => t._targetSignature?._depthStencilFormat)
            .filter((f): f is string => !!f);

    it("defaults to depth24plus-stencil8, so existing scenes are unchanged", () => {
        const { surface } = makeSurface(1);
        const scene = createSceneContext(surface);
        expect(depthFormatsOf(scene)).toEqual(["depth24plus-stencil8"]);
    });

    it("uses the requested format for the scene's depth target", () => {
        const { surface } = makeSurface(1);
        const scene = createSceneContext(surface, { depthFormat: "depth32float" });
        expect(depthFormatsOf(scene)).toEqual(["depth32float"]);
    });

    it("applies to the MSAA path too, where the colour target owns depth", () => {
        // MSAA routes depth onto the colour RT instead of a separate one. Missing
        // that branch would silently leave multisampled scenes on 24-bit depth.
        const { surface } = makeSurface(4);
        const scene = createSceneContext(surface, { depthFormat: "depth32float" });
        expect(depthFormatsOf(scene)).toEqual(["depth32float"]);
    });
});
