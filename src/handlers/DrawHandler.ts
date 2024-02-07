/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { Bytes4, ClusterTrianglesLimit, TextureFormats } from "../constants.js";
import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { Renderer } from "../core/Renderer.js";
import { BindGroupHandler } from "./BindGroupHandler.js";
import { PipelineHandler } from "./PipelineHandler.js";
import { AttachmentHandler } from "./AttachmentHandler.js";

export class DrawHandler {
    private readonly renderer: Renderer;

    public readonly indirectData: Uint32Array;
    public indirectBuffer: Nullable<GPUBuffer>;
    private readbackBuffer: Nullable<GPUBuffer>;
    private bundle: Nullable<GPURenderBundle>;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.indirectData = new Uint32Array([
            ClusterTrianglesLimit * 3,
            0,
            0,
            0,
        ]);
        this.indirectBuffer = null;
        this.readbackBuffer = null;
        this.bundle = null;
    }

    public prepare(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.indirectBuffer = this.createIndirectBuffer(device);
        this.readbackBuffer = this.createReadbackBuffer(device);
    }

    private createIndirectBuffer(device: GPUDevice): GPUBuffer {
        const buffer: GPUBuffer = device.createBuffer({
            label: "draw-indirect-buffer",
            size: this.indirectData.byteLength,
            usage:
                GPUBufferUsage.INDIRECT |
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC |
                GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(buffer, 0, this.indirectData);
        return buffer;
    }

    private createReadbackBuffer(device: GPUDevice): GPUBuffer {
        assert(this.indirectBuffer);
        return device.createBuffer({
            size: 1 * Bytes4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }

    public encode(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.bundle = this.encodeBundle(device);
    }

    private encodeBundle(device: GPUDevice): GPURenderBundle {
        const pipelines: PipelineHandler = this.renderer.handlers.pipeline;
        const bindGroups: BindGroupHandler = this.renderer.handlers.bindGroup;
        assert(pipelines.draw && bindGroups.draw && this.indirectBuffer);
        const encoder: GPURenderBundleEncoder =
            device.createRenderBundleEncoder({
                label: "draw-render-bundle",
                colorFormats: [this.renderer.presentationFormat],
                depthStencilFormat: TextureFormats.depth,
            } as GPURenderBundleEncoderDescriptor);
        encoder.setPipeline(pipelines.draw);
        encoder.setBindGroup(0, bindGroups.draw);
        encoder.drawIndirect(this.indirectBuffer, 0);
        return encoder.finish();
    }

    public setExecute(instanceCount: int): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device && this.indirectBuffer);
        this.indirectData[1] = instanceCount;
        device.queue.writeBuffer(
            this.indirectBuffer,
            1 * Bytes4,
            this.indirectData,
            1,
            1,
        );
    }

    public execute(encoder: GPUCommandEncoder): void {
        const attachments: AttachmentHandler =
            this.renderer.handlers.attachment;
        assert(attachments.color && attachments.depthStencil && this.bundle);
        attachments.update();
        const renderPass: GPURenderPassEncoder = encoder.beginRenderPass({
            label: "draw-render-pass",
            colorAttachments: [attachments.color],
            depthStencilAttachment: attachments.depthStencil,
            timestampWrites: this.renderer.analytics.getDrawPassTimestamps(),
        } as GPURenderPassDescriptor);
        renderPass.executeBundles([this.bundle]);
        renderPass.end();
    }

    public resolve(encoder: GPUCommandEncoder): void {
        assert(this.readbackBuffer && this.indirectBuffer);
        if (this.readbackBuffer.mapState !== "unmapped") {
            return;
        }
        encoder.copyBufferToBuffer(
            this.indirectBuffer,
            1 * Bytes4,
            this.readbackBuffer,
            0,
            1 * Bytes4,
        );
    }

    public readback(callback: (instanceCount: int) => void): void {
        assert(this.readbackBuffer);
        if (this.readbackBuffer.mapState !== "unmapped") {
            return;
        }
        this.readbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
            assert(this.readbackBuffer);
            const result: Uint32Array = new Uint32Array(
                this.readbackBuffer.getMappedRange().slice(0),
            );
            this.readbackBuffer.unmap();
            callback(result[0]);
        });
    }
}
