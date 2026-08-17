/**
 * `createTexture2DFromPixels`'s optional mip chain.
 *
 * The three things this covers are the three ways a mip chain silently does
 * nothing: allocating one level and calling it a chain, allocating a chain the
 * blit cannot write into (no RENDER_ATTACHMENT), and allocating a chain the
 * sampler never reads (WebGPU's default `mipmapFilter` is "nearest", which picks
 * one level instead of blending two). Each of those renders exactly like the
 * unmipped texture, so none of them shows up as an error.
 */
import { describe, expect, it } from "vitest";
import { createTexture2DFromPixels } from "../../../packages/babylon-lite/src/texture/pixels-texture";
import { rebuildTexture2D } from "../../../packages/babylon-lite/src/texture/texture-recovery";
import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import type { Texture2D } from "../../../packages/babylon-lite/src/texture/texture-2d";

interface Captured {
    createDesc?: GPUTextureDescriptor;
    samplerDesc?: GPUSamplerDescriptor;
    writeCalls: number;
    /** Render passes begun — one per generated mip level. */
    passes: number;
}

function newCap(): Captured {
    return { writeCalls: 0, passes: 0 };
}

function makeEngine(cap: Captured): EngineContext {
    const device = {
        createTexture: (desc: GPUTextureDescriptor) => {
            cap.createDesc = desc;
            return {
                format: desc.format,
                mipLevelCount: desc.mipLevelCount ?? 1,
                createView: () => ({ _kind: "view" }),
                destroy: () => undefined,
            } as unknown as GPUTexture;
        },
        createSampler: (desc: GPUSamplerDescriptor) => ((cap.samplerDesc = desc), { _kind: "sampler" } as unknown as GPUSampler),
        // Enough of a device for `generateMipmaps` to record its blit chain.
        createShaderModule: () => ({ _kind: "shader" }),
        createBindGroupLayout: () => ({ _kind: "bgl" }),
        createBindGroup: () => ({ _kind: "bg" }),
        createPipelineLayout: () => ({ _kind: "layout" }),
        createRenderPipeline: () => ({ _kind: "pipeline" }),
        createCommandEncoder: () => ({
            beginRenderPass: () => {
                cap.passes++;
                return { setPipeline: () => undefined, setBindGroup: () => undefined, draw: () => undefined, end: () => undefined };
            },
            finish: () => ({ _kind: "commands" }),
        }),
        queue: {
            writeTexture: () => {
                cap.writeCalls++;
            },
            submit: () => undefined,
        },
    };
    return { _device: device as unknown as GPUDevice } as unknown as EngineContext;
}

const PIXELS = new Uint8Array(64 * 64 * 4);

describe("createTexture2DFromPixels mip chains", () => {
    it("allocates a single level and a nearest mipmap filter by default", () => {
        const cap = newCap();
        createTexture2DFromPixels(makeEngine(cap), PIXELS, 64, 64);

        expect(cap.createDesc?.mipLevelCount).toBe(1);
        expect(cap.samplerDesc?.mipmapFilter).toBe("nearest");
    });

    it("does not widen usage for callers that want no chain", () => {
        // RENDER_ATTACHMENT on every lookup table and pixel-art sprite would be a
        // cost paid by the majority for the minority.
        const cap = newCap();
        createTexture2DFromPixels(makeEngine(cap), PIXELS, 64, 64);

        expect(Number(cap.createDesc?.usage) & GPUTextureUsage.RENDER_ATTACHMENT).toBe(0);
    });

    it("allocates the full chain, renderable and trilinearly sampled, when asked", () => {
        const cap = newCap();
        createTexture2DFromPixels(makeEngine(cap), PIXELS, 64, 64, { mipMaps: true, minFilter: "linear", magFilter: "linear" });

        // 64 -> 1 is seven levels, and the whole chain matters: a sprite minified
        // to a couple of pixels reads from the small end of it.
        expect(cap.createDesc?.mipLevelCount).toBe(7);
        expect(Number(cap.createDesc?.usage) & GPUTextureUsage.RENDER_ATTACHMENT).not.toBe(0);
        expect(cap.samplerDesc?.mipmapFilter).toBe("linear");
    });

    it("sizes the chain from the LONGER side, so a non-square texture reaches 1x1", () => {
        const cap = newCap();
        createTexture2DFromPixels(makeEngine(cap), new Uint8Array(256 * 4 * 4), 256, 4, { mipMaps: true });

        expect(cap.createDesc?.mipLevelCount).toBe(9);
    });

    it("leaves the levels EMPTY, because the caller owns when they are filled", () => {
        // `updateTexture2DFromPixels` rewrites level 0 and cannot know to rebuild
        // the rest, so a chain filled at creation would go stale on the first
        // update — invisibly, since stale mips only show under minification.
        const cap = newCap();
        createTexture2DFromPixels(makeEngine(cap), PIXELS, 64, 64, { mipMaps: true });

        expect(cap.writeCalls).toBe(1);
        expect(cap.passes, "no blit chain was recorded at creation").toBe(0);
    });
});

describe("device-lost recovery of a mipped pixel texture", () => {
    it("restores the chain AND fills it, having no caller to defer to", async () => {
        const cap = newCap();
        const engine = makeEngine(cap);
        const tex: Texture2D = {
            texture: undefined as unknown as GPUTexture,
            view: undefined as unknown as GPUTextureView,
            sampler: undefined as unknown as GPUSampler,
            width: 64,
            height: 64,
            _recoverySource: {
                kind: "pixels",
                data: PIXELS,
                width: 64,
                height: 64,
                options: { mipMaps: true, minFilter: "linear", magFilter: "linear" },
            },
        } as unknown as Texture2D;

        await rebuildTexture2D(engine, tex);

        expect(cap.createDesc?.mipLevelCount).toBe(7);
        expect(cap.samplerDesc?.mipmapFilter).toBe("linear");
        // Six blits for seven levels. A recovered texture that came back with an
        // allocated-but-empty chain would be a device loss that quietly degrades
        // sampling rather than recovering from it.
        expect(cap.passes).toBe(6);
    });

    it("still restores an unmipped pixel texture without touching the blit path", async () => {
        const cap = newCap();
        const engine = makeEngine(cap);
        const tex: Texture2D = {
            texture: undefined as unknown as GPUTexture,
            view: undefined as unknown as GPUTextureView,
            sampler: undefined as unknown as GPUSampler,
            width: 64,
            height: 64,
            _recoverySource: { kind: "pixels", data: PIXELS, width: 64, height: 64, options: {} },
        } as unknown as Texture2D;

        await rebuildTexture2D(engine, tex);

        expect(cap.createDesc?.mipLevelCount).toBe(1);
        expect(cap.passes).toBe(0);
    });
});
