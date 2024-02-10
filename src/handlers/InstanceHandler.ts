/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { Renderer } from "../components/Renderer.js";
import { BindGroupHandler } from "./BindGroupHandler.js";
import { PipelineHandler } from "./PipelineHandler.js";
import {
    Bytes4,
    DrawPairLayout,
    QueueHeaderLayout,
    QueueLimit,
} from "../constants.js";

export class InstanceHandler {
    private readonly renderer: Renderer;

    public queueBuffer: Nullable<GPUBuffer>;

    public readonly indirectData: Uint32Array;
    public indirectBuffer: Nullable<GPUBuffer>;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.queueBuffer = null;
        this.indirectData = new Uint32Array([0, 1, 1]);
        this.indirectBuffer = null;
    }

    public prepare(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.queueBuffer = device.createBuffer({
            label: "queue-buffer",
            size: (QueueLimit * DrawPairLayout + QueueHeaderLayout) * Bytes4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        this.indirectBuffer = this.createIndirectBuffer(device);
    }

    private createIndirectBuffer(device: GPUDevice): GPUBuffer {
        const buffer: GPUBuffer = device.createBuffer({
            label: "instance-indirect-buffer",
            size: this.indirectData.byteLength,
            usage:
                GPUBufferUsage.INDIRECT |
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(buffer, 0, this.indirectData);
        return buffer;
    }

    public setExecute(workgroupDispatchCount: int): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device && this.indirectBuffer);
        this.indirectData[0] = workgroupDispatchCount;
        device.queue.writeBuffer(
            this.indirectBuffer,
            0,
            this.indirectData,
            0,
            1,
        );
    }

    public resetQueue(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device && this.queueBuffer);
        device.queue.writeBuffer(
            this.queueBuffer,
            0,
            new Uint32Array([0]),
            0,
            1,
        );
    }

    public execute(encoder: GPUCommandEncoder): void {
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        const bindGroups: BindGroupHandler = this.renderer.handlers.bindGroup;
        assert(pipelines.instance && bindGroups.instance);
        assert(this.indirectBuffer);
        const pass: GPUComputePassEncoder = encoder.beginComputePass({
            label: "instance-compute-pass",
            timestampWrites:
                this.renderer.analytics.getInstancePassTimestamps(),
        } as GPUComputePassDescriptor);
        pass.setPipeline(pipelines.instance);
        pass.setBindGroup(0, bindGroups.instance);
        pass.dispatchWorkgroupsIndirect(this.indirectBuffer, 0);
        pass.end();
    }
}
