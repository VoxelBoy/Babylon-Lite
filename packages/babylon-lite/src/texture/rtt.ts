/**
 * Render-to-texture helper — eager allocation of a render target's GPU
 * textures so the color attachment, or depth attachment for depth-only targets,
 * can be exposed as a sampled texture BEFORE the frame graph is built.
 */

import type { EngineContext } from "../engine/engine.js";
import { getBilinearSampler, getNearestSampler } from "../resource/samplers.js";
import { getOrCreateSampler } from "../resource/gpu-pool.js";
import type { RenderTarget, RenderTargetDescriptor } from "../engine/render-target.js";
import { createRenderTarget, buildRenderTarget } from "../engine/render-target.js";
import type { Texture2D } from "./texture-2d.js";

/** How a render target's color attachment should be sampled once it is exposed
 *  as a texture. Omit the whole object for the historical default: a bilinear,
 *  clamp-to-edge, single-level sampler.
 *
 *  Defaults are deliberately not "whatever the texture supports". A target
 *  allocated with `mips: true` still samples from level 0 alone unless
 *  `mipmapFilter` is set, and repeat addressing is never inferred — a tiling
 *  target is a decision the caller makes, and guessing it wrong turns a tiled
 *  surface into a stretched edge texel with nothing in the log. */
export interface RenderTargetTextureSampling {
    /** U address mode. Default `"clamp-to-edge"`. Use `"repeat"` for a target that tiles. */
    addressModeU?: GPUAddressMode;
    /** V address mode. Default `"clamp-to-edge"`. */
    addressModeV?: GPUAddressMode;
    /** Minification filter. Default `"linear"`. */
    minFilter?: GPUFilterMode;
    /** Magnification filter. Default `"linear"`. */
    magFilter?: GPUFilterMode;
    /** Mip filter. Default unset (level 0 only). Requires the target's `mips: true`. */
    mipmapFilter?: GPUMipmapFilterMode;
    /** Max anisotropy. Default `1`. WebGPU requires linear min/mag/mip filters above 1. */
    maxAnisotropy?: number;
}

/** Eagerly allocate a render target's GPU textures and return a sampled-texture
 *  view of the color attachment, or the depth attachment for depth-only targets.
 *  Marks the RT so `buildRenderTarget` won't realloc.
 *
 *  The descriptor's size MUST be fixed (not `"canvas"`) because the canvas size
 *  may change before the frame graph builds, which would invalidate the eagerly-
 *  created texture handle that downstream bind groups have already captured. */
export function createRenderTargetTexture(
    engine: EngineContext,
    descriptor: RenderTargetDescriptor,
    sampling?: RenderTargetTextureSampling
): { rt: RenderTarget; texture: Texture2D } {
    if ("canvas" in descriptor.size) {
        throw new Error(
            "createRenderTargetTexture: descriptor.size must be fixed { width, height } pixels, not a SurfaceContext (would invalidate eagerly-allocated textures when the canvas resizes)."
        );
    }
    const fixedSize = descriptor.size;
    const rt = createRenderTarget(descriptor);
    buildRenderTarget(rt, engine);
    rt._eager = true;
    if (!rt._colorTexture || !rt._colorView) {
        if (!rt._depthTexture) {
            throw new Error("createRenderTargetTexture: render target has no color or depth texture (no format / depthStencilFormat?).");
        }
        const texture: Texture2D = {
            texture: rt._depthTexture,
            view: rt._depthTexture.createView({ aspect: "depth-only" }),
            sampler: getNearestSampler(engine),
            width: fixedSize.width,
            height: fixedSize.height,
            invertY: false,
            _sampleType: "depth",
        };
        return { rt, texture };
    }
    if (sampling?.mipmapFilter && !descriptor.mips) {
        throw new Error("createRenderTargetTexture: sampling.mipmapFilter requires the descriptor's mips: true — there would be no levels to filter between.");
    }
    // `rt._colorView` is the render-pass attachment, which names a single mip
    // level. Sampling needs the whole chain, so build a separate full view.
    const view = descriptor.mips ? rt._colorTexture.createView() : rt._colorView;
    const texture: Texture2D = {
        texture: rt._colorTexture,
        view,
        sampler: sampling
            ? getOrCreateSampler(engine, {
                  addressModeU: sampling.addressModeU ?? "clamp-to-edge",
                  addressModeV: sampling.addressModeV ?? "clamp-to-edge",
                  minFilter: sampling.minFilter ?? "linear",
                  magFilter: sampling.magFilter ?? "linear",
                  mipmapFilter: sampling.mipmapFilter,
                  maxAnisotropy: sampling.maxAnisotropy ?? 1,
              })
            : getBilinearSampler(engine),
        width: fixedSize.width,
        height: fixedSize.height,
        invertY: true,
    };
    return { rt, texture };
}
