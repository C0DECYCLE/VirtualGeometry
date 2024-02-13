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
import { InstanceHandler } from "./InstanceHandler.js";
import { EntityHandler } from "./EntityHandler.js";

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
        const entity: EntityHandler = this.renderer.handlers.entity;
        const instance: InstanceHandler = this.renderer.handlers.instance;
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        const cluster: ClusterHandler = this.renderer.handlers.cluster;
        assert(pipelines.instance && instance.queueBuffer);
        assert(entity.buffer && cluster.indirectBuffer);
        return device.createBindGroup({
            label: "instance-bindgroup",
            layout: pipelines.instance.getBindGroupLayout(0),
            entries: [
                this.createBinding(0, entity.buffer),
                this.createBinding(1, instance.queueBuffer),
                this.createBinding(2, cluster.indirectBuffer),
            ],
        } as GPUBindGroupDescriptor);
    }

    private createClusterBindGroup(device: GPUDevice): GPUBindGroup {
        const uniforms: UniformHandler = this.renderer.handlers.uniform;
        const instance: InstanceHandler = this.renderer.handlers.instance;
        const geometries: GeometryHandler = this.renderer.handlers.geometry;
        const draw: DrawHandler = this.renderer.handlers.draw;
        const cluster: ClusterHandler = this.renderer.handlers.cluster;
        const entity: EntityHandler = this.renderer.handlers.entity;
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        assert(pipelines.cluster && draw.indirectBuffer && uniforms.buffer);
        assert(geometries.clustersBuffer && cluster.drawPairBuffer);
        assert(instance.queueBuffer && entity.buffer);
        return device.createBindGroup({
            label: "cluster-bindgroup",
            layout: pipelines.cluster.getBindGroupLayout(0),
            entries: [
                this.createBinding(0, uniforms.buffer),
                this.createBinding(1, instance.queueBuffer),
                this.createBinding(2, geometries.clustersBuffer),
                this.createBinding(3, entity.buffer),
                this.createBinding(4, draw.indirectBuffer),
                this.createBinding(5, cluster.drawPairBuffer),
            ],
        } as GPUBindGroupDescriptor);
    }

    private createDrawBindGroup(device: GPUDevice): GPUBindGroup {
        const uniforms: UniformHandler = this.renderer.handlers.uniform;
        const geometries: GeometryHandler = this.renderer.handlers.geometry;
        const entity: EntityHandler = this.renderer.handlers.entity;
        const cluster: ClusterHandler = this.renderer.handlers.cluster;
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        assert(uniforms.buffer && pipelines.draw && cluster.drawPairBuffer);
        assert(geometries.trianglesBuffer && geometries.verticesBuffer);
        assert(entity.buffer);
        return device.createBindGroup({
            label: "draw-bindgroup",
            layout: pipelines.draw.getBindGroupLayout(0),
            entries: [
                this.createBinding(0, uniforms.buffer),
                this.createBinding(1, cluster.drawPairBuffer),
                this.createBinding(2, entity.buffer),
                this.createBinding(3, geometries.trianglesBuffer),
                this.createBinding(4, geometries.verticesBuffer),
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
