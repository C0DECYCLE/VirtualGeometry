/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import {
    AnalyticSamples,
    ClusterTrianglesLimit,
    DrawPairLimit,
} from "../constants.js";
import { DrawHandler } from "../handlers/DrawHandler.js";
import { EntityHandler } from "../handlers/EntityHandler.js";
import { RollingAverage } from "../utilities/RollingAverage.js";
import { warn } from "../utilities/logger.js";
import { assert, dotit } from "../utilities/utils.js";
import { Nullable, float, int } from "../utils.type.js";
import { GPUTiming } from "./GPUTiming.js";
import { Stats } from "./Stats.js";

export class Analytics {
    private readonly deltas: {
        frame: RollingAverage;
        cpu: RollingAverage;
        gpuInstance: RollingAverage;
        gpuCluster: RollingAverage;
        gpuDraw: RollingAverage;
    };
    private readonly stats: Stats;
    private timings: Nullable<{
        gpuInstance: GPUTiming;
        gpuCluster: GPUTiming;
        gpuDraw: GPUTiming;
    }>;

    public constructor() {
        this.deltas = {
            frame: new RollingAverage(AnalyticSamples),
            cpu: new RollingAverage(AnalyticSamples),
            gpuInstance: new RollingAverage(AnalyticSamples),
            gpuCluster: new RollingAverage(AnalyticSamples),
            gpuDraw: new RollingAverage(AnalyticSamples),
        };
        this.stats = this.createStats();
        this.timings = null;
    }

    private createStats(): Stats {
        const stats: Stats = new Stats();
        stats.set("frame delta", 0);
        stats.set("clusters post", 0);
        return stats;
    }

    public prepare(device: GPUDevice): void {
        this.timings = {
            gpuInstance: new GPUTiming(device),
            gpuCluster: new GPUTiming(device),
            gpuDraw: new GPUTiming(device),
        };
        this.stats.show();
    }

    public getInstancePassTimestamps(): GPUComputePassTimestampWrites {
        assert(this.timings);
        return this.timings.gpuInstance.timestampWrites;
    }

    public getClusterPassTimestamps(): GPUComputePassTimestampWrites {
        assert(this.timings);
        return this.timings.gpuCluster.timestampWrites;
    }

    public getDrawPassTimestamps(): GPURenderPassTimestampWrites {
        assert(this.timings);
        return this.timings.gpuDraw.timestampWrites;
    }

    public preFrame(): void {
        this.stats.time("cpu delta");
    }

    public resolve(encoder: GPUCommandEncoder): void {
        assert(this.timings);
        this.timings.gpuInstance.resolve(encoder);
        this.timings.gpuCluster.resolve(encoder);
        this.timings.gpuDraw.resolve(encoder);
    }

    public postFrame(
        now: float,
        entities: EntityHandler,
        draws: DrawHandler,
    ): void {
        const deltas = this.deltas;
        const stats = this.stats;
        const deltaToFps = this.deltaToFps;
        assert(this.timings);

        stats.time("cpu delta", "cpu delta");
        deltas.cpu.sample(stats.get("cpu delta")!);

        stats.set("frame delta", now - stats.get("frame delta")!);
        deltas.frame.sample(stats.get("frame delta")!);

        const gpuSum: float =
            deltas.gpuInstance.get() +
            deltas.gpuCluster.get() +
            deltas.gpuDraw.get();
        this.timings.gpuInstance.readback((ms: float) =>
            deltas.gpuInstance.sample(ms),
        );
        this.timings.gpuCluster.readback((ms: float) =>
            deltas.gpuCluster.sample(ms),
        );
        this.timings.gpuDraw.readback((ms: float) => deltas.gpuDraw.sample(ms));

        stats.set("instances pre", entities.count() * 1);
        // instances post

        // clusters pre
        draws.readback((instanceCount: int) => {
            if (instanceCount > DrawPairLimit) {
                warn("DrawHandler: Tried to draw more clusters than limit.");
            }
            stats.set("clusters post", instanceCount * 1);
        });

        stats.set("triangles pre", entities.countLeaveTriangles() * 1);
        stats.set(
            "triangles post",
            stats.get("clusters post")! * ClusterTrianglesLimit,
        );

        stats.set("vertices pre", stats.get("triangles pre")! * 3);
        stats.set("vertices post", stats.get("triangles post")! * 3);

        stats.update(`
            <b>frame rate: ${deltaToFps(deltas.frame.get())} fps</b><br>
            frame delta: ${dotit(deltas.frame.get().toFixed(2))} ms<br>
            <br>
            <b>cpu rate: ${deltaToFps(deltas.cpu.get())} fps</b><br>
            cpu delta: ${dotit(deltas.cpu.get().toFixed(2))} ms<br>
            <br>
            <b>gpu rate: ${deltaToFps(gpuSum)} fps</b><br>
            gpu delta: ${dotit(gpuSum.toFixed(2))} ms<br>
            | instance: ${dotit(deltas.gpuInstance.get().toFixed(2))} ms<br>
            | cluster: ${dotit(deltas.gpuCluster.get().toFixed(2))} ms<br>
            | draw: ${dotit(deltas.gpuDraw.get().toFixed(2))} ms<br>
            <br>
            <b>instances:</b><br>
            | pre: ${dotit(stats.get("instances pre")!)}<br>
            | post: ? // count of ones passed the culling<br>
            <br>
            <b>clusters:</b><br>
            | pre: ? // sum of clusters of each passed instance<br>
            | post: ${dotit(stats.get("clusters post")!)}<br>
            <br>
            <b>triangles:</b><br>
            | pre: ${dotit(stats.get("triangles pre")!)}<br>
            | post: ${dotit(stats.get("triangles post")!)}<br>
            <br>
            <b>vertices:</b><br>
            | pre: ${dotit(stats.get("vertices pre")!)}<br> 
            | post: ${dotit(stats.get("vertices post")!)}<br>
        `);

        stats.set("frame delta", now);
    }

    private deltaToFps(ms: float): string {
        return dotit(1_000 / ms);
    }
}
