/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, December 2023
 */

import { float, int } from "../../types/utils.type.js";
import { Bytes4 } from "../constants.js";

export class GPUTiming {
    private static readonly Capacity: int = 2;

    private readonly querySet: GPUQuerySet;
    private readonly queryBuffer: GPUBuffer;
    private readonly readbackBuffer: GPUBuffer;
    public readonly timestampWrites:
        | GPURenderPassTimestampWrites
        | GPUComputePassTimestampWrites;

    public constructor(device: GPUDevice) {
        this.querySet = this.createSet(device);
        this.queryBuffer = this.createBuffer(device);
        this.readbackBuffer = this.createReadback(device);
        this.timestampWrites = {
            querySet: this.querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
        } as GPURenderPassTimestampWrites | GPUComputePassTimestampWrites;
    }

    private createSet(device: GPUDevice): GPUQuerySet {
        return device.createQuerySet({
            type: "timestamp",
            count: GPUTiming.Capacity,
        } as GPUQuerySetDescriptor);
    }

    private createBuffer(device: GPUDevice): GPUBuffer {
        return device.createBuffer({
            size: GPUTiming.Capacity * (Bytes4 * 2), //64bit
            usage:
                GPUBufferUsage.QUERY_RESOLVE |
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC |
                GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
    }

    private createReadback(device: GPUDevice): GPUBuffer {
        return device.createBuffer({
            size: this.queryBuffer.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }

    public resolve(encoder: GPUCommandEncoder): void {
        encoder.resolveQuerySet(
            this.querySet,
            0,
            GPUTiming.Capacity,
            this.queryBuffer,
            0,
        );
        if (this.readbackBuffer.mapState !== "unmapped") {
            return;
        }
        encoder.copyBufferToBuffer(
            this.queryBuffer,
            0,
            this.readbackBuffer,
            0,
            this.readbackBuffer.size,
        );
    }

    public readback(callback: (ms: float) => void): void {
        if (this.readbackBuffer.mapState !== "unmapped") {
            return;
        }
        this.readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const nanos: BigInt64Array = new BigInt64Array(
                this.readbackBuffer.getMappedRange().slice(0),
            );
            this.readbackBuffer.unmap();
            const nanoDelta: float = Number(nanos[1] - nanos[0]);
            callback(nanoDelta / 1_000_000);
        });
    }
}
