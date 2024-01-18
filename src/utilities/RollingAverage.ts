/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float } from "./utils.type.js";

export class RollingAverage {
    private readonly sampleLength: int;
    private readonly samples: float[];

    private total: float = 0;
    private cursor: float = 0;

    public constructor(sampleLength: int = 30) {
        this.samples = [];
        this.sampleLength = sampleLength;
    }

    public sample(value: float): void {
        this.total += value - (this.samples[this.cursor] || 0);
        this.samples[this.cursor] = value;
        this.cursor = (this.cursor + 1) % this.sampleLength;
    }

    public get(): float {
        return this.total / this.samples.length;
    }
}
