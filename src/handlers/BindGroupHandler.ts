/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { Renderer } from "../core/Renderer.js";
import { UniformHandler } from "./UniformHandler.js";
import { GeometryHandler } from "./GeometryHandler.js";
import { PipelineHandler } from "./PipelineHandler.js";
import { DrawHandler } from "./DrawHandler.js";
import { ClusterHandler } from "./ClusterHandler.js";

export class BindGroupHandler {
    private readonly renderer: Renderer;

    public instance: Nullable<GPUBindGroup>;
    public cluster: Nullable<GPUBindGroup>;
    public draw: Nullable<GPUBindGroup>;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.instance = null;
        this.cluster = null;
        this.draw = null;
    }

    public prepare(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.instance = this.createInstanceBindGroup(device);
        this.cluster = this.createClusterBindGroup(device);
        this.draw = this.createDrawBindGroup(device);
    }

    private createInstanceBindGroup(device: GPUDevice): GPUBindGroup {
        const uniforms: UniformHandler = this.renderer.handlers.uniform;
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        assert(pipelines.instance && uniforms.buffer);
        return device.createBindGroup({
            label: "instance-bindgroup",
            layout: pipelines.instance.getBindGroupLayout(0),
            entries: [this.createBinding(0, uniforms.buffer)],
        } as GPUBindGroupDescriptor);
    }

    private createClusterBindGroup(device: GPUDevice): GPUBindGroup {
        const uniforms: UniformHandler = this.renderer.handlers.uniform;
        const geometries: GeometryHandler = this.renderer.handlers.geometry;
        const cluster: ClusterHandler = this.renderer.handlers.cluster;
        const draw: DrawHandler = this.renderer.handlers.draw;
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        assert(pipelines.cluster && draw.indirectBuffer && uniforms.buffer);
        assert(geometries.clustersBuffer && cluster.clusterDrawBuffer);
        return device.createBindGroup({
            label: "cluster-bindgroup",
            layout: pipelines.cluster.getBindGroupLayout(0),
            entries: [
                this.createBinding(0, uniforms.buffer),
                this.createBinding(1, geometries.clustersBuffer),
                this.createBinding(2, cluster.clusterDrawBuffer),
                this.createBinding(3, draw.indirectBuffer),
            ],
        } as GPUBindGroupDescriptor);
    }

    private createDrawBindGroup(device: GPUDevice): GPUBindGroup {
        const uniforms: UniformHandler = this.renderer.handlers.uniform;
        const geometries: GeometryHandler = this.renderer.handlers.geometry;
        const cluster: ClusterHandler = this.renderer.handlers.cluster;
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        assert(uniforms.buffer && pipelines.draw && cluster.clusterDrawBuffer);
        assert(geometries.trianglesBuffer && geometries.verticesBuffer);
        return device.createBindGroup({
            label: "draw-bindgroup",
            layout: pipelines.draw.getBindGroupLayout(0),
            entries: [
                this.createBinding(0, uniforms.buffer),
                this.createBinding(1, cluster.clusterDrawBuffer),
                this.createBinding(2, geometries.trianglesBuffer),
                this.createBinding(3, geometries.verticesBuffer),
            ],
        } as GPUBindGroupDescriptor);
    }

    private createBinding(index: int, buffer: GPUBuffer): GPUBindGroupEntry {
        return {
            binding: index,
            resource: { buffer: buffer } as GPUBindingResource,
        } as GPUBindGroupEntry;
    }
}
