import { describe, expect, it } from "vitest";

import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import { buildRenderTarget, createRenderTarget } from "../../../packages/babylon-lite/src/engine/render-target";
import { createRenderTargetTexture } from "../../../packages/babylon-lite/src/texture/rtt";
import { clearSamplerCache } from "../../../packages/babylon-lite/src/resource/gpu-pool";

interface Capture {
    textures: GPUTextureDescriptor[];
    views: (GPUTextureViewDescriptor | undefined)[];
    samplers: GPUSamplerDescriptor[];
}

function makeEngine(): { engine: EngineContext; capture: Capture } {
    const capture: Capture = { textures: [], views: [], samplers: [] };
    const device = {
        createTexture: (d: GPUTextureDescriptor) => {
            capture.textures.push(d);
            return {
                descriptor: d,
                format: d.format,
                mipLevelCount: d.mipLevelCount ?? 1,
                createView: (v?: GPUTextureViewDescriptor) => {
                    capture.views.push(v);
                    return {} as GPUTextureView;
                },
                destroy: () => undefined,
            } as unknown as GPUTexture;
        },
        createSampler: (d: GPUSamplerDescriptor) => {
            capture.samplers.push(d);
            return d as unknown as GPUSampler;
        },
    } as unknown as GPUDevice;
    const engine = { _device: device } as unknown as EngineContext;
    // The sampler pool is keyed by descriptor and lives across tests; clear it
    // so a cache hit from an earlier case cannot mask a missing createSampler.
    clearSamplerCache(engine);
    return { engine, capture };
}

const SIZE = { width: 512, height: 512 };

describe("render target mip chains", () => {
    it("allocates a single level by default", () => {
        const { engine, capture } = makeEngine();
        const rt = createRenderTarget({ format: "rgba8unorm", samples: 1, size: SIZE });
        buildRenderTarget(rt, engine);
        expect(capture.textures[0]!.mipLevelCount).toBe(1);
    });

    it("allocates a full chain when mips is set", () => {
        const { engine, capture } = makeEngine();
        const rt = createRenderTarget({ format: "rgba8unorm", samples: 1, size: SIZE, mips: true });
        buildRenderTarget(rt, engine);
        // 512 -> 1 is ten levels.
        expect(capture.textures[0]!.mipLevelCount).toBe(10);
    });

    it("restricts the attachment view to level 0 when mipped", () => {
        const { engine, capture } = makeEngine();
        const rt = createRenderTarget({ format: "rgba8unorm", samples: 1, size: SIZE, mips: true });
        buildRenderTarget(rt, engine);
        // A render-pass attachment must name exactly one level, so the
        // attachment view cannot be the default full-chain view.
        expect(capture.views[0]).toEqual({ baseMipLevel: 0, mipLevelCount: 1 });
    });

    it("rejects mips combined with MSAA rather than failing later in WebGPU", () => {
        const { engine } = makeEngine();
        const rt = createRenderTarget({ format: "rgba8unorm", samples: 4, size: SIZE, mips: true });
        expect(() => buildRenderTarget(rt, engine)).toThrow(/mutually exclusive/);
    });
});

describe("createRenderTargetTexture sampling", () => {
    it("keeps the historical bilinear clamp sampler when no sampling is given", () => {
        const { engine, capture } = makeEngine();
        createRenderTargetTexture(engine, { format: "rgba8unorm", samples: 1, size: SIZE });
        expect(capture.samplers).toEqual([{ magFilter: "linear", minFilter: "linear" }]);
    });

    it("applies repeat addressing, mip filtering and anisotropy when asked", () => {
        const { engine, capture } = makeEngine();
        createRenderTargetTexture(
            engine,
            { format: "rgba8unorm", samples: 1, size: SIZE, mips: true },
            { addressModeU: "repeat", addressModeV: "repeat", mipmapFilter: "linear", maxAnisotropy: 16 }
        );
        expect(capture.samplers).toEqual([
            {
                addressModeU: "repeat",
                addressModeV: "repeat",
                minFilter: "linear",
                magFilter: "linear",
                mipmapFilter: "linear",
                maxAnisotropy: 16,
            },
        ]);
    });

    it("samples through the full mip chain, not the level-0 attachment view", () => {
        const { engine, capture } = makeEngine();
        createRenderTargetTexture(engine, { format: "rgba8unorm", samples: 1, size: SIZE, mips: true }, { mipmapFilter: "linear" });
        // First view is the level-0 attachment, second is the sampled full chain.
        expect(capture.views[0]).toEqual({ baseMipLevel: 0, mipLevelCount: 1 });
        expect(capture.views[1]).toBeUndefined();
    });

    it("rejects mip filtering on a target that has no mip chain", () => {
        const { engine } = makeEngine();
        expect(() => createRenderTargetTexture(engine, { format: "rgba8unorm", samples: 1, size: SIZE }, { mipmapFilter: "linear" })).toThrow(/mips: true/);
    });
});
