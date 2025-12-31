/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { Renderer } from "../components/Renderer.js";
import { BindGroupHandler } from "./BindGroupHandler.js";
import { PipelineHandler } from "./PipelineHandler.js";
import { DrawPairLimit, Bytes4, DrawPairLayout } from "../constants.js";

export class ClusterHandler {
    private readonly renderer: Renderer;

    public drawPairBuffer: Nullable<GPUBuffer>;

    private dispatchCount: int;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.drawPairBuffer = null;
        this.dispatchCount = 1;
    }

    public prepare(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.drawPairBuffer = device.createBuffer({
            label: "draw-pair-buffer",
            size: DrawPairLimit * DrawPairLayout * Bytes4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
    }

    public setExecute(workgroupDispatchCount: int): void {
        this.dispatchCount = workgroupDispatchCount;
    }

    public execute(encoder: GPUCommandEncoder): void {
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        const bindGroups: BindGroupHandler = this.renderer.handlers.bindGroup;
        assert(pipelines.cluster && bindGroups.cluster);
        const pass: GPUComputePassEncoder = encoder.beginComputePass({
            label: "cluster-compute-pass",
            timestampWrites: this.renderer.analytics.getClusterPassTimestamps(),
        } as GPUComputePassDescriptor);
        pass.setPipeline(pipelines.cluster);
        pass.setBindGroup(0, bindGroups.cluster);
        pass.dispatchWorkgroups(this.dispatchCount);
        pass.end();
    }
}
