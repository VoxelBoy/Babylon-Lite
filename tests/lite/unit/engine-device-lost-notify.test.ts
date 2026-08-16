import { describe, expect, it, vi } from "vitest";

import { createEngine } from "../../../packages/babylon-lite/src/engine/engine";

/**
 * `EngineOptions.onDeviceLost` is notification only. An application whose
 * GPU-resident state cannot be rebuilt coherently must be able to learn that the
 * device went away WITHOUT Lite trying to replace it — and without reaching past
 * the public API for `engine._device.lost`.
 */
function makeGpu(): { lose: (info: GPUDeviceLostInfo) => void } {
    let resolveLost: (info: GPUDeviceLostInfo) => void = () => {};
    const lost = new Promise<GPUDeviceLostInfo>((r) => {
        resolveLost = r;
    });
    const device = {
        lost,
        features: new Set<string>(),
        limits: {},
        createBuffer: vi.fn(() => ({ destroy: vi.fn() }) as unknown as GPUBuffer),
        createTexture: vi.fn(() => ({ createView: vi.fn(), destroy: vi.fn() }) as unknown as GPUTexture),
        createBindGroupLayout: vi.fn(),
        createPipelineLayout: vi.fn(),
        createShaderModule: vi.fn(),
        createRenderPipeline: vi.fn(),
        createSampler: vi.fn(),
        queue: { writeBuffer: vi.fn(), submit: vi.fn() },
        destroy: vi.fn(),
    } as unknown as GPUDevice;
    const adapter = {
        features: new Set<string>(),
        limits: {},
        requestDevice: vi.fn(async () => device),
    } as unknown as GPUAdapter;
    vi.stubGlobal("navigator", { gpu: { requestAdapter: vi.fn(async () => adapter), getPreferredCanvasFormat: () => "bgra8unorm" } });
    return { lose: (info) => resolveLost(info) };
}

const canvas = () =>
    ({
        width: 64,
        height: 64,
        clientWidth: 64,
        clientHeight: 64,
        setAttribute: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getContext: () => ({
            configure: vi.fn(),
            unconfigure: vi.fn(),
            getCurrentTexture: vi.fn(() => ({ width: 64, height: 64, createView: vi.fn(() => ({})), destroy: vi.fn() })),
        }),
    }) as unknown as HTMLCanvasElement;

describe("EngineOptions.onDeviceLost", () => {
    it("reports a lost device without enabling any recovery", async () => {
        const { lose } = makeGpu();
        const seen: GPUDeviceLostInfo[] = [];
        const engine = await createEngine(canvas(), { onDeviceLost: (info) => seen.push(info) });
        expect(seen).toHaveLength(0);

        const info = { reason: "unknown", message: "the GPU went away" } as GPUDeviceLostInfo;
        lose(info);
        await Promise.resolve();
        await Promise.resolve();

        expect(seen).toEqual([info]);
        // Nothing was rebuilt: no replacement device was requested.
        expect((navigator.gpu.requestAdapter as unknown as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(1);
        expect(engine).toBeTruthy();
    });

    it("reports a `destroyed` loss too, rather than deciding for the caller", async () => {
        const { lose } = makeGpu();
        const seen: GPUDeviceLostInfo[] = [];
        await createEngine(canvas(), { onDeviceLost: (info) => seen.push(info) });

        // Teardown normally reports "destroyed". Filtering it here would hide a
        // real loss that happened to report the same reason, so the caller
        // decides, not the engine.
        const info = { reason: "destroyed", message: "" } as GPUDeviceLostInfo;
        lose(info);
        await Promise.resolve();
        await Promise.resolve();

        expect(seen).toEqual([info]);
    });

    it("is inert when not supplied", async () => {
        const { lose } = makeGpu();
        await createEngine(canvas());
        lose({ reason: "unknown", message: "" } as GPUDeviceLostInfo);
        await Promise.resolve();
        // Nothing to assert but the absence of a throw: an unhandled rejection
        // or a call into undefined would fail the test.
        expect(true).toBe(true);
    });
});
