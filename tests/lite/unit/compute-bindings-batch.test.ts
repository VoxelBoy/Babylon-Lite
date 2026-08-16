import { describe, expect, it, vi } from "vitest";

import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import {
    createComputeShader,
    createComputeBindings,
    setComputeBindingsStorageBuffer,
    setComputeStorageBuffer,
    disposeComputeBindings,
    dispatchCompute,
    dispatchComputeBatch,
} from "../../../packages/babylon-lite/src/compute/compute-shader";
import { createStorageBuffer } from "../../../packages/babylon-lite/src/resource/storage-buffer";

interface Recorder {
    submits: number;
    encoders: number;
    setPipeline: ReturnType<typeof vi.fn>;
    setBindGroup: ReturnType<typeof vi.fn>;
    dispatchWorkgroups: ReturnType<typeof vi.fn>;
    createComputePipeline: ReturnType<typeof vi.fn>;
    createBindGroup: ReturnType<typeof vi.fn>;
}

function makeEngine(): { engine: EngineContext; rec: Recorder } {
    const rec: Partial<Recorder> = { submits: 0, encoders: 0 };
    rec.setPipeline = vi.fn();
    rec.setBindGroup = vi.fn();
    rec.dispatchWorkgroups = vi.fn();
    rec.createComputePipeline = vi.fn((d: GPUComputePipelineDescriptor) => ({ label: d.label }) as unknown as GPUComputePipeline);
    rec.createBindGroup = vi.fn((d: GPUBindGroupDescriptor) => d as unknown as GPUBindGroup);
    const pass = {
        setPipeline: rec.setPipeline,
        setBindGroup: rec.setBindGroup,
        dispatchWorkgroups: rec.dispatchWorkgroups,
        end: vi.fn(),
    } as unknown as GPUComputePassEncoder;
    const device = {
        createShaderModule: vi.fn((d: GPUShaderModuleDescriptor) => d as unknown as GPUShaderModule),
        createBindGroupLayout: vi.fn((d: GPUBindGroupLayoutDescriptor) => d as unknown as GPUBindGroupLayout),
        createPipelineLayout: vi.fn((d: GPUPipelineLayoutDescriptor) => d as unknown as GPUPipelineLayout),
        createComputePipeline: rec.createComputePipeline,
        createBindGroup: rec.createBindGroup,
        createBuffer: vi.fn((d: GPUBufferDescriptor) => {
            const backing = new ArrayBuffer(Number(d.size));
            return { label: d.label, size: Number(d.size), getMappedRange: () => backing, unmap: vi.fn(), destroy: vi.fn() } as unknown as GPUBuffer;
        }),
        createCommandEncoder: vi.fn(() => {
            rec.encoders!++;
            return { beginComputePass: () => pass, finish: () => ({}) as GPUCommandBuffer } as unknown as GPUCommandEncoder;
        }),
        queue: {
            writeBuffer: vi.fn(),
            submit: vi.fn(() => {
                rec.submits!++;
            }),
        },
    } as unknown as GPUDevice;
    const engine = { _device: device, _disposables: [] } as unknown as EngineContext;
    return { engine, rec: rec as Recorder };
}

const SOURCE = `@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) { out[gid.x] = params[gid.x]; }`;

function makeShader(engine: EngineContext) {
    return createComputeShader(engine, {
        name: "fill",
        computeSource: SOURCE,
        storageBuffers: [
            { name: "params", type: "array<f32>" },
            { name: "out", type: "array<f32>", writable: true },
        ],
    });
}

describe("ComputeBindings", () => {
    it("shares one pipeline across many binding sets", () => {
        const { engine, rec } = makeEngine();
        const shader = makeShader(engine);
        const slab = createStorageBuffer(engine, 256, { writable: true });
        const a = createComputeBindings(shader, "bodyA");
        const b = createComputeBindings(shader, "bodyB");
        for (const [set, label] of [
            [a, "a"],
            [b, "b"],
        ] as const) {
            setComputeBindingsStorageBuffer(set, "params", createStorageBuffer(engine, new Float32Array(4), { label }));
            setComputeBindingsStorageBuffer(set, "out", slab);
        }

        dispatchComputeBatch(engine, [
            { shader, bindings: a, x: 1, y: 2 },
            { shader, bindings: b, x: 1, y: 3 },
        ]);

        // One pipeline for identical WGSL — the whole point of separating the
        // program from the data it runs over.
        expect(rec.createComputePipeline).toHaveBeenCalledTimes(1);
        expect(rec.setPipeline).toHaveBeenCalledTimes(1);
        expect(rec.setBindGroup).toHaveBeenCalledTimes(2);
        expect(rec.setBindGroup.mock.calls[0]![1]).not.toBe(rec.setBindGroup.mock.calls[1]![1]);
    });

    it("records the whole batch into one command buffer and submits once", () => {
        const { engine, rec } = makeEngine();
        const shader = makeShader(engine);
        const slab = createStorageBuffer(engine, 256, { writable: true });
        const sets = ["a", "b", "c"].map((label) => {
            const set = createComputeBindings(shader, label);
            setComputeBindingsStorageBuffer(set, "params", createStorageBuffer(engine, new Float32Array(4), { label }));
            setComputeBindingsStorageBuffer(set, "out", slab);
            return set;
        });

        dispatchComputeBatch(
            engine,
            sets.map((bindings, i) => ({ shader, bindings, x: 4, y: i + 1 }))
        );

        expect(rec.encoders).toBe(1);
        expect(rec.submits).toBe(1);
        expect(rec.dispatchWorkgroups.mock.calls).toEqual([
            [4, 1, 1],
            [4, 2, 1],
            [4, 3, 1],
        ]);
    });

    it("still submits once per call for the unbatched path, so the comparison is real", () => {
        const { engine, rec } = makeEngine();
        const shader = makeShader(engine);
        setComputeStorageBuffer(shader, "params", createStorageBuffer(engine, new Float32Array(4)));
        setComputeStorageBuffer(shader, "out", createStorageBuffer(engine, 256, { writable: true }));
        dispatchCompute(engine, shader, 1);
        dispatchCompute(engine, shader, 1);
        expect(rec.submits).toBe(2);
    });

    it("caches a binding set's bind group until a buffer changes", () => {
        const { engine, rec } = makeEngine();
        const shader = makeShader(engine);
        const slab = createStorageBuffer(engine, 256, { writable: true });
        const set = createComputeBindings(shader);
        setComputeBindingsStorageBuffer(set, "params", createStorageBuffer(engine, new Float32Array(4)));
        setComputeBindingsStorageBuffer(set, "out", slab);

        dispatchComputeBatch(engine, [{ shader, bindings: set, x: 1 }]);
        const afterFirst = rec.createBindGroup.mock.calls.length;
        dispatchComputeBatch(engine, [{ shader, bindings: set, x: 1 }]);
        expect(rec.createBindGroup.mock.calls.length).toBe(afterFirst);

        setComputeBindingsStorageBuffer(set, "params", createStorageBuffer(engine, new Float32Array(4)));
        dispatchComputeBatch(engine, [{ shader, bindings: set, x: 1 }]);
        expect(rec.createBindGroup.mock.calls.length).toBe(afterFirst + 1);
    });

    it("rejects a binding set built for a different program", () => {
        const { engine } = makeEngine();
        const first = makeShader(engine);
        const second = makeShader(engine);
        const set = createComputeBindings(first);
        expect(() => dispatchComputeBatch(engine, [{ shader: second, bindings: set, x: 1 }])).toThrow(/different ComputeShader/);
    });

    it("names the missing binding rather than dispatching a partial set", () => {
        const { engine } = makeEngine();
        const shader = makeShader(engine);
        const set = createComputeBindings(shader, "half-bound");
        setComputeBindingsStorageBuffer(set, "params", createStorageBuffer(engine, new Float32Array(4)));
        expect(() => dispatchComputeBatch(engine, [{ shader, bindings: set, x: 1 }])).toThrow(/"out" was declared but never bound/);
    });

    it("rejects a read-only allocation on a read_write binding", () => {
        const { engine } = makeEngine();
        const shader = makeShader(engine);
        const set = createComputeBindings(shader);
        expect(() => setComputeBindingsStorageBuffer(set, "out", createStorageBuffer(engine, 256))).toThrow(/writable: true/);
    });

    it("refuses to use a disposed binding set", () => {
        const { engine } = makeEngine();
        const shader = makeShader(engine);
        const set = createComputeBindings(shader);
        setComputeBindingsStorageBuffer(set, "params", createStorageBuffer(engine, new Float32Array(4)));
        setComputeBindingsStorageBuffer(set, "out", createStorageBuffer(engine, 256, { writable: true }));
        disposeComputeBindings(set);
        expect(() => dispatchComputeBatch(engine, [{ shader, bindings: set, x: 1 }])).toThrow(/has been disposed/);
    });

    it("is a no-op for an empty batch", () => {
        const { engine, rec } = makeEngine();
        dispatchComputeBatch(engine, []);
        expect(rec.encoders).toBe(0);
        expect(rec.submits).toBe(0);
    });
});
