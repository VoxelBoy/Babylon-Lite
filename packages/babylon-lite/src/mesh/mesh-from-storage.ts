/**
 * Module: mesh-from-storage
 *
 * Geometry that lives only on the GPU. A mesh created here sources its vertex
 * stream directly from a `StorageBuffer` — so a compute pass can produce vertices
 * and the draw can consume them in place, with **no readback and no copy**.
 *
 * This is the seam that procedural/GPU-generated worlds need. Lite's canonical
 * path (`createMeshFromData`) takes CPU `Float32Array`s and uploads them, which
 * forces GPU-produced geometry through a device→host→device round trip.
 *
 * Zero-cost when unused: nothing in the core render path imports this module, so
 * it tree-shakes away entirely for scenes that never call the factory.
 *
 * Usage contract:
 *  - The allocation must come from `createStorageBuffer(..., { writable: true, vertex: true })`.
 *  - The material must describe the packing via `ShaderMaterialOptions.vertexLayout`
 *    (its `arrayStride`/`format` must match what the compute shader writes).
 *  - Bounds are the caller's responsibility: the CPU never sees these vertices, so
 *    `boundMin`/`boundMax` must be supplied (analytically, or from a known envelope)
 *    for frustum culling to stay correct.
 *  - CPU-side picking (`_cpuPositions`) is unavailable by construction.
 */
import type { EngineContext } from "../engine/engine.js";
import type { Mesh } from "./mesh.js";
import { initMeshTransform } from "./mesh.js";
import { BU } from "../engine/gpu-flags.js";
import { createMappedBuffer } from "../resource/gpu-buffers.js";
import { _getStorageBufferHandle, type StorageBuffer } from "../resource/storage-buffer.js";

/** Describes a mesh whose vertices are produced on the GPU. */
export interface MeshFromStorageOptions {
    /** Vertex source. Must be `writable: true, vertex: true`. */
    readonly storage: StorageBuffer;
    /** Triangle indices. Uploaded once; index topology is static per chunk. */
    readonly indices: Uint32Array;
    /** Number of vertices addressed by `indices`, used for validation only. */
    readonly vertexCount: number;
    /** Byte stride of one vertex inside `storage`. Must match the material's `vertexLayout`. */
    readonly arrayStride: number;
    /** First vertex of this mesh within a shared allocation.
     *
     *  This is how many meshes share ONE slab: each takes a slot and addresses it
     *  through the draw call's `baseVertex`, rather than a non-zero `setVertexBuffer`
     *  bind offset (which corrupts vertex fetch on some AMD/Dawn paths). Default 0. */
    readonly baseVertex?: number;
    /** Analytic lower bound of the produced geometry, in mesh-local space. */
    readonly boundMin?: readonly [number, number, number];
    /** Analytic upper bound of the produced geometry, in mesh-local space. */
    readonly boundMax?: readonly [number, number, number];
}

/** Create a mesh that draws straight from a GPU storage allocation. */
export function createMeshFromStorageBuffer(engine: EngineContext, name: string, options: MeshFromStorageOptions): Mesh {
    const { storage, indices, vertexCount, arrayStride, baseVertex = 0 } = options;

    if (!storage._vertex) {
        throw new Error("createMeshFromStorageBuffer: storage must be created with { vertex: true } so it carries GPUBufferUsage.VERTEX.");
    }
    if (!Number.isInteger(arrayStride) || arrayStride <= 0 || arrayStride % 4 !== 0) {
        throw new Error(`createMeshFromStorageBuffer: arrayStride must be a positive multiple of 4, received ${arrayStride}.`);
    }
    const required = (baseVertex + vertexCount) * arrayStride;
    if (required > storage.byteLength) {
        throw new Error(`createMeshFromStorageBuffer: slot needs ${required} bytes but the allocation is ${storage.byteLength} bytes.`);
    }

    const vertexBuffer = _getStorageBufferHandle(engine, storage);
    const indexBuffer = createMappedBuffer(engine, indices, BU.INDEX, `${name}-indices`);

    const mesh = initMeshTransform({
        name,
        material: null as unknown as Mesh["material"],
        receiveShadows: false,
        boundMin: options.boundMin ? [...options.boundMin] : undefined,
        boundMax: options.boundMax ? [...options.boundMax] : undefined,
        _gpu: {
            // Every attribute reads the one shared allocation; the material's
            // `vertexLayout` splits it into fields via stride + offset.
            positionBuffer: vertexBuffer,
            normalBuffer: vertexBuffer,
            uvBuffer: vertexBuffer,
            indexBuffer,
            indexCount: indices.length,
            indexFormat: "uint32",
            _baseVertex: baseVertex,
        },
    });

    return mesh;
}
