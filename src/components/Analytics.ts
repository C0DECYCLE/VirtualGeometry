/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { AnalyticSamples } from "../constants.js";
import { RollingAverage } from "../utilities/RollingAverage.js";
import { assert } from "../utilities/utils.js";
import { Nullable, float, int } from "../utils.type.js";
import { GPUTiming } from "./GPUTiming.js";
import { Stats } from "./Stats.js";

export class Analytics {
    private readonly deltas: {
        frame: RollingAverage;
        cpu: RollingAverage;
        gpu: RollingAverage;
    };
    private readonly stats: Stats;
    private timings: Nullable<{
        gpu: GPUTiming;
    }>;

    public constructor() {
        this.deltas = {
            frame: new RollingAverage(AnalyticSamples),
            cpu: new RollingAverage(AnalyticSamples),
            gpu: new RollingAverage(AnalyticSamples),
        };
        this.stats = this.createStats();
        this.timings = null;
    }

    private createStats(): Stats {
        const stats: Stats = new Stats();
        const deltaKeys: string[] = Object.keys(this.deltas);
        for (let i: int = 0; i < deltaKeys.length; i++) {
            stats.set(`${deltaKeys[i]} delta`, 0);
        }
        return stats;
    }

    public prepare(device: GPUDevice): void {
        this.timings = {
            gpu: new GPUTiming(device),
        };
        this.stats.show();
    }

    public getRenderPassTimestamps(): GPURenderPassTimestampWrites {
        assert(this.timings);
        return this.timings.gpu.timestampWrites;
    }

    public preFrame(): void {
        this.stats.time("cpu delta");
    }

    public resolve(encoder: GPUCommandEncoder): void {
        assert(this.timings);
        this.timings.gpu.resolve(encoder);
    }

    public postFrame(now: float): void {
        this.stats.time("cpu delta", "cpu delta");
        this.deltas.cpu.sample(this.stats.get("cpu delta")!);
        this.stats.set("frame delta", now - this.stats.get("frame delta")!);
        this.deltas.frame.sample(this.stats.get("frame delta")!);
        assert(this.timings);
        this.timings.gpu.readback((ms: float) => {
            this.stats.set("gpu delta", ms);
            this.deltas.gpu.sample(ms);
        });
        // prettier-ignore
        this.stats.update(`
            <b>frame rate: ${(1_000 / this.deltas.frame.get()).toFixed(0)} fps</b><br>
            frame delta: ${this.deltas.frame.get().toFixed(2)} ms<br>
            <br>
            <b>cpu rate: ${(1_000 / this.deltas.cpu.get()).toFixed(0)} fps</b><br>
            cpu delta: ${this.deltas.cpu.get().toFixed(2)} ms<br>
            <br>
            <b>gpu rate: ${(1_000 / this.deltas.cpu.get()).toFixed(0)} fps</b><br>
            gpu delta: ${this.deltas.cpu.get().toFixed(2)} ms<br>
        `);
        this.stats.set("frame delta", now);
    }
}
