import { describe, expect, it } from "vitest";

import type { EngineContext } from "../../../packages/babylon-lite/src/engine/engine";
import { addTask, removeTask } from "../../../packages/babylon-lite/src/frame-graph/frame-graph-actions";
import { createFrameGraph } from "../../../packages/babylon-lite/src/frame-graph/frame-graph";
import type { Pass } from "../../../packages/babylon-lite/src/frame-graph/pass";
import type { Task } from "../../../packages/babylon-lite/src/frame-graph/task";

function makeTask(name: string, executions: string[], disposals: string[]): Task {
    const task = {
        name,
        engine: {} as EngineContext,
        _passes: [] as Pass[],
        record(): void {},
        execute(): number {
            executions.push(name);
            return 1;
        },
        dispose(): void {
            disposals.push(name);
        },
    } satisfies Task;
    return task;
}

describe("removeTask", () => {
    it("stops executing the removed task and leaves the rest in order", () => {
        const executions: string[] = [];
        const disposals: string[] = [];
        const graph = createFrameGraph({} as EngineContext);
        const first = makeTask("first", executions, disposals);
        const bake = makeTask("bake", executions, disposals);
        const last = makeTask("last", executions, disposals);
        addTask(graph, first);
        addTask(graph, bake);
        addTask(graph, last);
        graph.build();

        expect(graph.execute()).toBe(3);
        expect(executions).toEqual(["first", "bake", "last"]);

        // The motivating case: a one-shot bake retires itself after its frame.
        executions.length = 0;
        expect(removeTask(graph, bake)).toBe(true);

        expect(graph.execute()).toBe(2);
        expect(executions).toEqual(["first", "last"]);
    });

    it("does not dispose the removed task — the caller still owns its resources", () => {
        const executions: string[] = [];
        const disposals: string[] = [];
        const graph = createFrameGraph({} as EngineContext);
        const bake = makeTask("bake", executions, disposals);
        addTask(graph, bake);
        graph.build();

        removeTask(graph, bake);
        expect(disposals).toEqual([]);

        // And the graph no longer disposes it either, so a bake's render target
        // survives for the material that is going to sample it.
        graph.dispose();
        expect(disposals).toEqual([]);
    });

    it("reports false for a task that is not in the graph", () => {
        const executions: string[] = [];
        const disposals: string[] = [];
        const graph = createFrameGraph({} as EngineContext);
        const present = makeTask("present", executions, disposals);
        const absent = makeTask("absent", executions, disposals);
        addTask(graph, present);

        expect(removeTask(graph, absent)).toBe(false);
        expect(removeTask(graph, present)).toBe(true);
        expect(removeTask(graph, present)).toBe(false);
    });

    it("accepts a SceneContext as well as a FrameGraph, like addTask", () => {
        const executions: string[] = [];
        const disposals: string[] = [];
        const graph = createFrameGraph({} as EngineContext);
        const scene = { _frameGraph: graph } as unknown as Parameters<typeof addTask>[0];
        const task = makeTask("scene-task", executions, disposals);
        addTask(scene, task);
        graph.build();
        graph.execute();
        expect(executions).toEqual(["scene-task"]);

        expect(removeTask(scene, task)).toBe(true);
        executions.length = 0;
        graph.execute();
        expect(executions).toEqual([]);
    });
});
