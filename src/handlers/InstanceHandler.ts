/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { Renderer } from "../components/Renderer.js";
import { BindGroupHandler } from "./BindGroupHandler.js";
import { PipelineHandler } from "./PipelineHandler.js";
import { Bytes4, QueueHeaderLayout, QueueLimit } from "../constants.js";

export class InstanceHandler {
    private readonly renderer: Renderer;

    public readonly queueHeader: Uint32Array;
    public queueBuffer: Nullable<GPUBuffer>;

    private dispatchCount: int;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.queueHeader = new Uint32Array(QueueHeaderLayout);
        this.queueBuffer = null;
        this.dispatchCount = 1;
    }

    public prepare(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.queueBuffer = device.createBuffer({
            label: "queue-buffer",
            size: (QueueHeaderLayout + QueueLimit) * Bytes4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
    }

    public setExecute(workgroupDispatchCount: int): void {
        this.dispatchCount = workgroupDispatchCount;
    }

    public resetQueueHeader(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device && this.queueBuffer);
        device.queue.writeBuffer(
            this.queueBuffer,
            0,
            this.queueHeader.buffer,
            0,
            this.queueHeader.byteLength,
        );
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
}
