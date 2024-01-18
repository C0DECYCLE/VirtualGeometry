/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, December 2023
 */

import { float, int } from "../../types/utils.type.js";
import { bytes4 } from "../helper.js";

export class GPUTiming {
    private static readonly capacity: int = 2;

    private readonly querySet: GPUQuerySet;
    private readonly queryBuffer: GPUBuffer;
    private readonly queryReadbackBuffer: GPUBuffer;
    public readonly timestampWrites: GPURenderPassTimestampWrites;

    public constructor(device: GPUDevice) {
        this.querySet = this.createSet(device);
        this.queryBuffer = this.createBuffer(device);
        this.queryReadbackBuffer = this.createReadback(device);
        this.timestampWrites = {
            querySet: this.querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
        } as GPURenderPassTimestampWrites;
    }

    private createSet(device: GPUDevice): GPUQuerySet {
        return device.createQuerySet({
            type: "timestamp",
            count: GPUTiming.capacity,
        } as GPUQuerySetDescriptor);
    }

    private createBuffer(device: GPUDevice): GPUBuffer {
        return device.createBuffer({
            size: GPUTiming.capacity * (bytes4 * 2), //64bit
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
            GPUTiming.capacity,
            this.queryBuffer,
            0,
        );
        if (this.queryReadbackBuffer.mapState !== "unmapped") {
            return;
        }
        encoder.copyBufferToBuffer(
            this.queryBuffer,
            0,
            this.queryReadbackBuffer,
            0,
            this.queryReadbackBuffer.size,
        );
    }

    public readback(callback: (ms: float) => void): void {
        if (this.queryReadbackBuffer.mapState !== "unmapped") {
            return;
        }
        this.queryReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const nanos: BigInt64Array = new BigInt64Array(
                this.queryReadbackBuffer.getMappedRange().slice(0),
            );
            this.queryReadbackBuffer.unmap();
            const nanoDelta: float = Number(nanos[1] - nanos[0]);
            callback(nanoDelta / 1_000_000);
        });
    }
}
