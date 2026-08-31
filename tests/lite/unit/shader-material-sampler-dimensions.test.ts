/**
 * A ShaderMaterial sampler emits the WGSL texture type and the bind-group-layout
 * view dimension that its declared `viewDimension` asks for — including "3d",
 * which is the only way to sample a `Texture3D` from a material (effects could
 * always bind one; materials collapsed every dimension to 2d/2d-array).
 */
import { describe, expect, it, vi } from "vitest";

import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import type { RenderTargetSignature } from "../../../packages/babylon-lite/src/engine/render-target";
import type { ShaderSamplerOption } from "../../../packages/babylon-lite/src/material/shader/shader-material";
import { createShaderMaterial } from "../../../packages/babylon-lite/src/material/shader/shader-material";
import { getOrCreateShaderPipeline, getOrCreateShaderPipelineBindings } from "../../../packages/babylon-lite/src/material/shader/shader-pipeline";
import { clearSceneBGLCache } from "../../../packages/babylon-lite/src/render/scene-helpers";

const signature = {
    _colorFormat: "rgba8unorm",
    _depthStencilFormat: "depth24plus",
    _sampleCount: 1,
} as RenderTargetSignature;

/** Builds the group-1 layout and the shader module for a material carrying `samplers`,
 *  and returns what the device was asked to create. */
function build(samplers: readonly ShaderSamplerOption[]) {
    clearSceneBGLCache();
    const createBindGroupLayout = vi.fn((d: GPUBindGroupLayoutDescriptor) => d as unknown as GPUBindGroupLayout);
    const createShaderModule = vi.fn((d: GPUShaderModuleDescriptor) => d as unknown as GPUShaderModule);
    const device = {
        createBindGroupLayout,
        createShaderModule,
        createPipelineLayout: vi.fn((d: GPUPipelineLayoutDescriptor) => d as unknown as GPUPipelineLayout),
        createRenderPipeline: vi.fn((d: GPURenderPipelineDescriptor) => d as unknown as GPURenderPipeline),
    } as unknown as GPUDevice;
    const engine = { _device: device } as unknown as EngineContext;

    const material = createShaderMaterial({
        vertexSource: "@vertex fn mainVertex(input: VertexInput) -> @builtin(position) vec4f { return vec4f(input.position, 1); }",
        fragmentSource: "@fragment fn mainFragment() -> @location(0) vec4f { return vec4f(1); }",
        attributes: ["position"],
        samplers,
    });
    const bindings = getOrCreateShaderPipelineBindings(engine, material);
    getOrCreateShaderPipeline(engine, signature, material, bindings);

    // The group-1 layout is the one labelled for this material; the scene BGL is created first.
    const group1 = createBindGroupLayout.mock.calls.map((c) => c[0]).find((d) => d.label === "shader-material-group1")!;
    return {
        textureEntries: [...group1.entries].filter((e) => "texture" in e && e.texture),
        wgsl: createShaderModule.mock.calls.map((c) => c[0].code).join("\n"),
    };
}

describe("ShaderMaterial sampler view dimensions", () => {
    it("emits texture_3d and a 3d layout entry for a volume sampler", () => {
        const { textureEntries, wgsl } = build([{ name: "noise", viewDimension: "3d" }]);
        expect(wgsl).toContain("var noise: texture_3d<f32>");
        expect(wgsl).toContain("var noiseSampler: sampler");
        expect(textureEntries[0]!.texture).toMatchObject({ sampleType: "float", viewDimension: "3d" });
    });

    it("still emits texture_2d_array and a 2d-array layout entry", () => {
        const { textureEntries, wgsl } = build([{ name: "atlas", viewDimension: "2d-array" }]);
        expect(wgsl).toContain("var atlas: texture_2d_array<f32>");
        expect(textureEntries[0]!.texture).toMatchObject({ viewDimension: "2d-array" });
    });

    it("still defaults to texture_2d and a 2d layout entry", () => {
        const { textureEntries, wgsl } = build(["albedo"]);
        expect(wgsl).toContain("var albedo: texture_2d<f32>");
        expect(textureEntries[0]!.texture).toMatchObject({ viewDimension: "2d" });
    });

    it("still emits a comparison depth sampler", () => {
        const { textureEntries, wgsl } = build([{ name: "csm", viewDimension: "2d-array", comparison: true }]);
        expect(wgsl).toContain("var csm: texture_depth_2d_array");
        expect(wgsl).toContain("var csmSampler: sampler_comparison");
        expect(textureEntries[0]!.texture).toMatchObject({ sampleType: "depth", viewDimension: "2d-array" });
    });

    it("rejects a depth or comparison volume sampler at compile time", () => {
        // WGSL has no texture_depth_3d. ShaderSampler3DDecl makes both pairings unrepresentable,
        // so these are type errors rather than a WebGPU validation failure at pipeline creation.
        // A runtime guard was measured instead and rejected: it cost 62 bytes in every
        // ShaderMaterial scene, plus 5 bytes in scenes with no ShaderMaterial at all, because a
        // new throw renumbers every later lite-error code and some cross a digit boundary.

        // @ts-expect-error -- comparison is false | undefined on a "3d" sampler
        const comparison: ShaderSamplerOption = { name: "bad", viewDimension: "3d", comparison: true };
        // @ts-expect-error -- sampleType excludes "depth" on a "3d" sampler
        const depth: ShaderSamplerOption = { name: "bad", viewDimension: "3d", sampleType: "depth" };

        expect([comparison, depth]).toHaveLength(2);
    });
});
