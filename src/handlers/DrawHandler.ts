/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterTrianglesLimit, TextureFormats } from "../constants.js";
import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { Renderer } from "../core/Renderer.js";
import { BindGroupHandler } from "./BindGroupHandler.js";
import { PipelineHandler } from "./PipelineHandler.js";
import { AttachmentHandler } from "./AttachmentHandler.js";

export class DrawHandler {
    private readonly renderer: Renderer;

    public readonly data: Uint32Array;
    public indirectBuffer: Nullable<GPUBuffer>;
    private bundle: Nullable<GPURenderBundle>;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.data = new Uint32Array([ClusterTrianglesLimit * 3, 0, 0, 0]);
        this.indirectBuffer = null;
        this.bundle = null;
    }

    public prepare(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.indirectBuffer = this.createIndirectBuffer(device);
    }

    public encode(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.bundle = this.encodeBundle(device);
    }

    private createIndirectBuffer(device: GPUDevice): GPUBuffer {
        return device.createBuffer({
            label: "indirect-buffer",
            size: this.data.byteLength,
            usage:
                GPUBufferUsage.INDIRECT |
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
    }

    private encodeBundle(device: GPUDevice): GPURenderBundle {
        const bindGroups: BindGroupHandler = this.renderer.handlers.bindGroup;
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        assert(pipelines.draw && bindGroups.draw && this.indirectBuffer);
        const encoder: GPURenderBundleEncoder =
            device.createRenderBundleEncoder({
                label: "render-bundle",
                colorFormats: [this.renderer.presentationFormat],
                depthStencilFormat: TextureFormats.depth,
            } as GPURenderBundleEncoderDescriptor);
        encoder.setPipeline(pipelines.draw);
        encoder.setBindGroup(0, bindGroups.draw);
        encoder.drawIndirect(this.indirectBuffer, 0);
        return encoder.finish();
    }

    public synchronize(instances: int): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device && this.indirectBuffer);
        this.data[1] = instances;
        device.queue.writeBuffer(this.indirectBuffer, 0, this.data);
        //here only the instances part!
    }

    public render(encoder: GPUCommandEncoder): void {
        const attachments: AttachmentHandler =
            this.renderer.handlers.attachment;
        assert(attachments.color && attachments.depthStencil && this.bundle);
        attachments.update();
        const renderPass: GPURenderPassEncoder = encoder.beginRenderPass({
            label: "render-pass",
            colorAttachments: [attachments.color],
            depthStencilAttachment: attachments.depthStencil,
            timestampWrites: this.renderer.analytics.getRenderPassTimestamps(),
        } as GPURenderPassDescriptor);
        renderPass.executeBundles([this.bundle]);
        renderPass.end();
    }
}
