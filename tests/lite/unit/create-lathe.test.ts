import { describe, expect, it } from "vitest";

import { createLatheData } from "../../../packages/babylon-lite/src/mesh/create-lathe";
import { createRibbonData } from "../../../packages/babylon-lite/src/mesh/create-ribbon";

const SHAPE = [
    { x: 0.2, y: 0, z: 0 },
    { x: 0.6, y: 1, z: 0 },
    { x: 0.3, y: 2, z: 0 },
];

/** The path array a correct lathe must hand its ribbon. */
function expectedPaths(tessellation: number, arc = 1, clip = 0, radius = 1) {
    const step = ((Math.PI * 2) / tessellation) * arc;
    const paths = [];
    for (let i = 0; i <= tessellation - clip; i++) {
        paths.push(SHAPE.map((p) => ({ x: Math.cos(i * step) * p.x * radius, y: p.y, z: Math.sin(i * step) * p.x * radius })));
    }
    return paths;
}

describe("createLatheData", () => {
    it("is exactly the ribbon of the revolved profile", () => {
        // A lathe IS a ribbon of rotated paths — that is how Babylon builds it,
        // and building it the same way is what makes vertex order, winding and
        // UVs agree for free rather than by coincidence.
        const lathe = createLatheData({ shape: SHAPE, tessellation: 12 });
        const ribbon = createRibbonData({ pathArray: expectedPaths(12), closeArray: true });
        expect(Array.from(lathe.positions)).toEqual(Array.from(ribbon.positions));
        expect(Array.from(lathe.indices)).toEqual(Array.from(ribbon.indices));
        expect(Array.from(lathe.uvs)).toEqual(Array.from(ribbon.uvs));
    });

    it("applies radius, arc and clip the way Babylon does", () => {
        expect(Array.from(createLatheData({ shape: SHAPE, tessellation: 12, radius: 3 }).positions)).toEqual(
            Array.from(createRibbonData({ pathArray: expectedPaths(12, 1, 0, 3), closeArray: true }).positions)
        );
        expect(Array.from(createLatheData({ shape: SHAPE, tessellation: 12, arc: 0.5 }).positions)).toEqual(
            Array.from(createRibbonData({ pathArray: expectedPaths(12, 0.5), closeArray: true }).positions)
        );
        // `clip` drops trailing steps while KEEPING full-turn spacing, so it is
        // not the same as reducing tessellation.
        const clipped = createLatheData({ shape: SHAPE, tessellation: 12, clip: 4 });
        const unclipped = createLatheData({ shape: SHAPE, tessellation: 12 });
        expect(clipped.positions.length).toBeLessThan(unclipped.positions.length);
        expect(Array.from(clipped.positions)).toEqual(Array.from(createRibbonData({ pathArray: expectedPaths(12, 1, 4), closeArray: true }).positions));
    });

    it("clamps an out-of-range arc to a full turn rather than erroring", () => {
        const full = Array.from(createLatheData({ shape: SHAPE, tessellation: 8 }).positions);
        for (const arc of [0, -1, 2]) {
            expect(Array.from(createLatheData({ shape: SHAPE, tessellation: 8, arc }).positions), `arc ${arc}`).toEqual(full);
        }
    });

    it("closes the surface by default and opens it on request", () => {
        const closed = createLatheData({ shape: SHAPE, tessellation: 8 });
        const open = createLatheData({ shape: SHAPE, tessellation: 8, closed: false });
        // Closing joins the last path back to the first, which adds triangles.
        expect(closed.indices.length).toBeGreaterThan(open.indices.length);
    });

    it("rejects a profile too short to revolve", () => {
        expect(() => createLatheData({ shape: [{ x: 1, y: 0, z: 0 }] })).toThrow(/at least two points/);
    });
});
