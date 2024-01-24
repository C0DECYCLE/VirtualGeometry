/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { Renderer } from "../core/Renderer.js";
import { UniformHandler } from "./UniformHandler.js";
import { GeometryHandler } from "./GeometryHandler.js";
import { PipelineHandler } from "./PipelineHandler.js";

export class BindGroupHandler {
    private readonly renderer: Renderer;

    public draw: Nullable<GPUBindGroup>;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.draw = null;
    }

    public prepare(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.draw = this.createDrawBindGroup(device);
    }

    private createDrawBindGroup(device: GPUDevice): GPUBindGroup {
        const uniforms: UniformHandler = this.renderer.handlers.uniform;
        const geometries: GeometryHandler = this.renderer.handlers.geometry;
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        assert(uniforms.buffer && pipelines.draw);
        assert(geometries.indicesBuffer && geometries.verticesBuffer);
        return device.createBindGroup({
            label: "draw-bindgroup",
            layout: pipelines.draw.getBindGroupLayout(0),
            entries: [
                this.createBinding(0, uniforms.buffer),
                this.createBinding(1, geometries.indicesBuffer),
                this.createBinding(2, geometries.verticesBuffer),
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
