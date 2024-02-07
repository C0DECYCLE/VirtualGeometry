/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { Renderer } from "../core/Renderer.js";
import { BindGroupHandler } from "./BindGroupHandler.js";
import { PipelineHandler } from "./PipelineHandler.js";
import { ClusterDrawLimit, Bytes4 } from "../constants.js";

export class ClusterHandler {
    private readonly renderer: Renderer;

    public clusterDrawBuffer: Nullable<GPUBuffer>;

    public readonly indirectData: Uint32Array;
    public indirectBuffer: Nullable<GPUBuffer>;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.clusterDrawBuffer = null;
        this.indirectData = new Uint32Array([0, 1, 1]);
        this.indirectBuffer = null;
    }

    public prepare(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.clusterDrawBuffer = device.createBuffer({
            label: "cluster-draw-buffer",
            size: ClusterDrawLimit * Bytes4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        this.indirectBuffer = this.createIndirectBuffer(device);
    }

    private createIndirectBuffer(device: GPUDevice): GPUBuffer {
        const buffer: GPUBuffer = device.createBuffer({
            label: "cluster-indirect-buffer",
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

    public execute(encoder: GPUCommandEncoder): void {
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        const bindGroups: BindGroupHandler = this.renderer.handlers.bindGroup;
        assert(pipelines.cluster && bindGroups.cluster && this.indirectBuffer);
        const pass: GPUComputePassEncoder = encoder.beginComputePass({
            label: "cluster-compute-pass",
            timestampWrites: this.renderer.analytics.getClusterPassTimestamps(),
        } as GPUComputePassDescriptor);
        pass.setPipeline(pipelines.cluster);
        pass.setBindGroup(0, bindGroups.cluster);
        pass.dispatchWorkgroupsIndirect(this.indirectBuffer, 0);
        pass.end();
    }
}
