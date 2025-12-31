/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { Renderer } from "../components/Renderer.js";
import { Bytes4, QueueHeaderLayout, QueueLimit } from "../constants.js";
import { log } from "../utilities/logger.js";
import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { BindGroupHandler } from "./BindGroupHandler.js";
import { PipelineHandler } from "./PipelineHandler.js";

export class InstanceHandler {
    private readonly renderer: Renderer;

    public readonly queueData: Uint32Array;
    public queueBuffer: Nullable<GPUBuffer>;
    private readbackBuffer: Nullable<GPUBuffer>;

    private dispatchCount: int;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.queueData = new Uint32Array(QueueHeaderLayout + QueueLimit);
        for (let i: int = 0; i < QueueLimit; i++) {
            this.queueData[QueueHeaderLayout + i] = 0xffffffff;
        }
        this.queueBuffer = null;
        this.readbackBuffer = null;
        this.dispatchCount = 1;
    }

    public prepare(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.queueBuffer = this.createQueueBuffer(device);
        this.readbackBuffer = this.createReadbackBuffer(device);
    }

    private createQueueBuffer(device: GPUDevice): GPUBuffer {
        const buffer: GPUBuffer = device.createBuffer({
            label: "queue-buffer",
            size: (QueueHeaderLayout + QueueLimit) * Bytes4,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST |
                GPUBufferUsage.COPY_SRC,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(buffer, 0, this.queueData.buffer);
        return buffer;
    }

    private createReadbackBuffer(device: GPUDevice): GPUBuffer {
        assert(this.queueBuffer);
        return device.createBuffer({
            size: this.queueBuffer.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }

    public setExecute(workgroupDispatchCount: int): void {
        this.dispatchCount = workgroupDispatchCount;
    }

    public resetQueue(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device && this.queueBuffer);
        device.queue.writeBuffer(this.queueBuffer, 0, this.queueData.buffer);
    }

    public execute(encoder: GPUCommandEncoder): void {
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        const bindGroups: BindGroupHandler = this.renderer.handlers.bindGroup;
        assert(pipelines.instance && bindGroups.instance);
        const pass: GPUComputePassEncoder = encoder.beginComputePass({
            label: "instance-compute-pass",
            timestampWrites:
                this.renderer.analytics.getInstancePassTimestamps(),
        } as GPUComputePassDescriptor);
        pass.setPipeline(pipelines.instance);
        pass.setBindGroup(0, bindGroups.instance);
        pass.dispatchWorkgroups(this.dispatchCount);
        pass.end();
    }

    public resolve(encoder: GPUCommandEncoder): void {
        assert(this.readbackBuffer && this.queueBuffer);
        if (this.readbackBuffer.mapState !== "unmapped") {
            return;
        }
        encoder.copyBufferToBuffer(this.queueBuffer, this.readbackBuffer);
    }

    public readback(): void {
        assert(this.readbackBuffer);
        if (this.readbackBuffer.mapState !== "unmapped") {
            return;
        }
        this.readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
            assert(this.readbackBuffer);
            const buffer: ArrayBuffer = this.readbackBuffer
                .getMappedRange()
                .slice(0);
            this.readbackBuffer.unmap();
            const resultUint: Uint32Array = new Uint32Array(buffer);
            const resultInt: Int32Array = new Int32Array(buffer);
            log("head: " + resultUint[32 * 1 - 1]);
            log("tail: " + resultUint[32 * 2 - 1]);
            log("count: " + resultInt[32 * 3 - 1]);
            const offset: int = 0;
            const length: int = 20;
            for (let i: int = offset; i < offset + length; i++) {
                const value: int = resultUint[QueueHeaderLayout + i];
                if (value === 0xffffffff) {
                    log("ring[" + i + "]: UNUSED");
                } else {
                    const entity: int = value >>> 16;
                    const cluster: int = value & 0xffff;
                    log("ring[" + i + "]: " + entity + " | " + cluster);
                }
            }
        });
    }
}
