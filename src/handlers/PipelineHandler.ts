/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { TextureFormats } from "../constants.js";
import { assert } from "../utilities/utils.js";
import { Nullable } from "../utils.type.js";
import { Renderer } from "../core/Renderer.js";

export class PipelineHandler {
    private readonly renderer: Renderer;

    public evaluation: Nullable<GPUComputePipeline>;
    public draw: Nullable<GPURenderPipeline>;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.evaluation = null;
        this.draw = null;
    }

    public async prepare(): Promise<void> {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.evaluation = await this.createEvaluationPipeline(device);
        this.draw = await this.createDrawPipeline(device);
    }

    private async createEvaluationPipeline(
        device: GPUDevice,
    ): Promise<GPUComputePipeline> {
        const shader: Nullable<GPUShaderModule> =
            this.renderer.handlers.shader.evaluationModule;
        assert(shader);
        return await device.createComputePipelineAsync({
            label: "evaluation-pipeline",
            layout: "auto",
            compute: {
                module: shader,
                entryPoint: "cs",
            } as GPUProgrammableStage,
        } as GPUComputePipelineDescriptor);
    }

    private async createDrawPipeline(
        device: GPUDevice,
    ): Promise<GPURenderPipeline> {
        const shader: Nullable<GPUShaderModule> =
            this.renderer.handlers.shader.drawModule;
        assert(shader);
        return await device.createRenderPipelineAsync({
            label: "draw-pipeline",
            layout: "auto",
            vertex: {
                module: shader,
                entryPoint: "vs",
            } as GPUVertexState,
            fragment: {
                module: shader,
                entryPoint: "fs",
                targets: [{ format: this.renderer.presentationFormat }],
            } as GPUFragmentState,
            primitive: {
                topology: "triangle-list",
                cullMode: "back",
            } as GPUPrimitiveState,
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: TextureFormats.depth,
            } as GPUDepthStencilState,
        } as GPURenderPipelineDescriptor);
    }
}
