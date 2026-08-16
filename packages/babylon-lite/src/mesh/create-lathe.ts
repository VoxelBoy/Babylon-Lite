import type { Vec3 } from "../math/types.js";
import type { RibbonData } from "./create-ribbon.js";
import { createRibbonData } from "./create-ribbon.js";

/**
 * CreateLathe — a surface of revolution, matching Babylon.js
 * `MeshBuilder.CreateLathe` defaults.
 *
 * A lathe is a ribbon: the profile is rotated into `tessellation` paths around
 * the Y axis and those paths become the ribbon's path array. That is exactly how
 * Babylon builds it, and building it the same way here means the two agree on
 * vertex order, winding and UVs for free rather than by coincidence.
 *
 * Options are the subset that the ribbon underneath can express. `cap`,
 * `invertUV` and `sideOrientation` are intentionally omitted: Lite's ribbon has
 * no cap or UV-inversion concept, so accepting those options and ignoring them
 * would be worse than not accepting them.
 */

/** Options for {@link createLatheData}. Subset of Babylon's CreateLathe. */
export interface LatheOptions {
    /** Profile to revolve, in the XoY plane. `x` is the radius at that point, `y` the height. */
    shape: readonly Vec3[];
    /** Uniform radius multiplier applied to every profile point. Default `1`. */
    radius?: number;
    /** Number of rotational steps. Default `64`. */
    tessellation?: number;
    /** Fraction of a full turn to sweep, in `(0, 1]`. Default `1`. */
    arc?: number;
    /** Trailing steps to omit, for a partial sweep that keeps full-turn spacing. Default `0`. */
    clip?: number;
    /** Join the last path back to the first. Default `true`. */
    closed?: boolean;
}

/** Generate indexed vertex data for a surface of revolution. */
export function createLatheData(options: LatheOptions): RibbonData {
    const shape = options.shape;
    if (!shape || shape.length < 2) {
        throw new Error(`createLatheData: shape needs at least two points, received ${shape?.length ?? 0}.`);
    }
    // Babylon clamps out-of-range arcs to a full turn rather than erroring.
    const arc = options.arc ? (options.arc <= 0 || options.arc > 1 ? 1.0 : options.arc) : 1.0;
    const closed = options.closed === undefined ? true : options.closed;
    const radius = options.radius ?? 1;
    const tessellation = options.tessellation ?? 64;
    const clip = options.clip ?? 0;
    const step = ((Math.PI * 2) / tessellation) * arc;

    const pathArray: Vec3[][] = [];
    for (let i = 0; i <= tessellation - clip; i++) {
        const cos = Math.cos(i * step);
        const sin = Math.sin(i * step);
        const path: Vec3[] = [];
        for (let p = 0; p < shape.length; p++) {
            const point = shape[p]!;
            path.push({ x: cos * point.x * radius, y: point.y, z: sin * point.x * radius });
        }
        pathArray.push(path);
    }

    return createRibbonData({ pathArray, closeArray: closed });
}
