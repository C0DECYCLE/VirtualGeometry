/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { AnalyticSamples } from "../constants.js";
import { DrawHandler } from "../handlers/DrawHandler.js";
import { RollingAverage } from "../utilities/RollingAverage.js";
import { assert, dotit } from "../utilities/utils.js";
import { Nullable, float, int } from "../utils.type.js";
import { GPUTiming } from "./GPUTiming.js";
import { Stats } from "./Stats.js";

export class Analytics {
    private readonly deltas: {
        frame: RollingAverage;
        cpu: RollingAverage;
        gpuCluster: RollingAverage;
        gpuDraw: RollingAverage;
    };
    private readonly stats: Stats;
    private timings: Nullable<{
        gpuCluster: GPUTiming;
        gpuDraw: GPUTiming;
    }>;

    public constructor() {
        this.deltas = {
            frame: new RollingAverage(AnalyticSamples),
            cpu: new RollingAverage(AnalyticSamples),
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
            gpuCluster: new GPUTiming(device),
            gpuDraw: new GPUTiming(device),
        };
        this.stats.show();
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
        this.timings.gpuCluster.resolve(encoder);
        this.timings.gpuDraw.resolve(encoder);
    }

    public postFrame(now: float, draws: DrawHandler): void {
        const deltas = this.deltas;
        const stats = this.stats;
        const deltaToFps = this.deltaToFps;
        assert(this.timings);

        stats.time("cpu delta", "cpu delta");
        deltas.cpu.sample(stats.get("cpu delta")!);

        stats.set("frame delta", now - stats.get("frame delta")!);
        deltas.frame.sample(stats.get("frame delta")!);

        const gpuSum: float = deltas.gpuCluster.get() + deltas.gpuDraw.get();
        this.timings.gpuCluster.readback((ms: float) =>
            deltas.gpuCluster.sample(ms),
        );
        this.timings.gpuDraw.readback((ms: float) => deltas.gpuDraw.sample(ms));

        draws.readback((instanceCount: int) =>
            stats.set("clusters post", instanceCount * 1),
        );

        stats.update(`
            <b>frame rate: ${deltaToFps(deltas.frame.get())} fps</b><br>
            frame delta: ${dotit(deltas.frame.get().toFixed(2))} ms<br>
            <br>
            <b>cpu rate: ${deltaToFps(deltas.cpu.get())} fps</b><br>
            cpu delta: ${dotit(deltas.cpu.get().toFixed(2))} ms<br>
            <br>
            <b>gpu rate: ${deltaToFps(gpuSum)} fps</b><br>
            gpu delta: ${dotit(gpuSum.toFixed(2))} ms<br>
            | instance: ? ms<br>
            | cluster: ${dotit(deltas.gpuCluster.get().toFixed(2))} ms<br>
            | draw: ${dotit(deltas.gpuDraw.get().toFixed(2))} ms<br>
            <br>
            <b>instances:</b><br>
            | pre: ? // all instances known to renderer<br>
            | post: ? // ones passed the culling<br>
            <br>
            <b>clusters:</b><br>
            | pre: ? // sum of clusters of each passed instance<br>
            | post: ${dotit(stats.get("clusters post")!)}<br>
            <br>
            <b>triangles:</b><br>
            | pre: ? // sum of leave triangles of all instance<br>
            | post: ${dotit(stats.get("clusters post")! * 128)}<br>
            <br>
            <b>vertices:</b><br>
            | pre: ? // sum of leave vertices of all instance<br> 
            | post: ${dotit(stats.get("clusters post")! * 128 * 3)}<br>
        `);

        stats.set("frame delta", now);
    }

    private deltaToFps(ms: float): string {
        return dotit(1_000 / ms);
    }
}
