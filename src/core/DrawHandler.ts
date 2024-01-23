/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterTrianglesLimit, TextureFormats } from "../constants.js";
import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { Renderer } from "./Renderer.js";

export class DrawHandler {
    private readonly renderer: Renderer;

    public readonly data: Uint32Array;
    private buffer: Nullable<GPUBuffer>;

    private textures: Nullable<{ depth: GPUTexture }>;
    private textureViews: Nullable<{ depth: GPUTextureView }>;

    private pipeline: Nullable<GPURenderPipeline>;
    private bindGroup: Nullable<GPUBindGroup>;

    private attachments: Nullable<{
        color: GPURenderPassColorAttachment;
        depthStencil: GPURenderPassDepthStencilAttachment;
    }>;

    private passDescriptor: Nullable<GPURenderPassDescriptor>;
    private bundle: Nullable<GPURenderBundle>;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.data = new Uint32Array([ClusterTrianglesLimit * 3, 0, 0, 0]);
        this.buffer = null;
        this.textures = null;
        this.textureViews = null;
        this.pipeline = null;
        this.bindGroup = null;
        this.attachments = null;
        this.passDescriptor = null;
        this.bundle = null;
    }

    public async prepare(): Promise<void> {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.buffer = device.createBuffer({
            label: "indirect-buffer",
            size: this.data.byteLength,
            usage:
                GPUBufferUsage.INDIRECT |
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        this.textures = this.createTextures();
        this.textureViews = this.createTextureViews();
        this.pipeline = await this.createPipeline();
        this.bindGroup = this.createBindGroup();
        this.attachments = this.createAttachments();
        this.passDescriptor = this.describePass();
        this.bundle = this.encodeBundle();
    }

    private createTextures(): { depth: GPUTexture } {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        const canvas: Nullable<HTMLCanvasElement> = this.renderer.canvas;
        assert(canvas);
        const depthTexture: GPUTexture = device.createTexture({
            label: "depth-texture",
            size: [canvas.width, canvas.height],
            format: TextureFormats.depth,
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING,
        } as GPUTextureDescriptor);
        return {
            depth: depthTexture,
        };
    }

    private createTextureViews(): { depth: GPUTextureView } {
        assert(this.textures);
        return {
            depth: this.textures.depth.createView(),
        };
    }

    private async createPipeline(): Promise<GPURenderPipeline> {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        const shader: Nullable<GPUShaderModule> =
            this.renderer.shaderHandler.drawModule;
        assert(shader);
        const presentationFormat: GPUTextureFormat =
            this.renderer.presentationFormat;
        return await device.createRenderPipelineAsync({
            label: "render-pipeline",
            layout: "auto",
            vertex: {
                module: shader,
                entryPoint: "vs",
            } as GPUVertexState,
            fragment: {
                module: shader,
                entryPoint: "fs",
                targets: [{ format: presentationFormat }],
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

    private createBindGroup(): GPUBindGroup {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        const uniforms: Nullable<GPUBuffer> =
            this.renderer.uniformsHandler.buffer;
        assert(uniforms);
        const indices: Nullable<GPUBuffer> =
            this.renderer.geometryHandler.indicesBuffer;
        assert(indices);
        const vertices: Nullable<GPUBuffer> =
            this.renderer.geometryHandler.verticesBuffer;
        assert(vertices && this.pipeline);
        return device.createBindGroup({
            label: "render-bindgroup",
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: uniforms } as GPUBindingResource,
                } as GPUBindGroupEntry,
                {
                    binding: 1,
                    resource: { buffer: indices } as GPUBindingResource,
                } as GPUBindGroupEntry,
                {
                    binding: 2,
                    resource: { buffer: vertices } as GPUBindingResource,
                } as GPUBindGroupEntry,
            ],
        } as GPUBindGroupDescriptor);
    }

    private createAttachments(): {
        color: GPURenderPassColorAttachment;
        depthStencil: GPURenderPassDepthStencilAttachment;
    } {
        const view: GPUTextureView = this.renderer.requestContextView();
        const colorAttachment: GPURenderPassColorAttachment = {
            label: "color attachment",
            view: view,
            clearValue: [0.3, 0.3, 0.3, 1],
            loadOp: "clear",
            storeOp: "store",
        } as GPURenderPassColorAttachment;
        assert(this.textureViews);
        const depthStencilAttachment: GPURenderPassDepthStencilAttachment = {
            label: "depth stencil attachment",
            view: this.textureViews.depth,
            depthClearValue: 1,
            depthLoadOp: "clear",
            depthStoreOp: "store",
        } as GPURenderPassDepthStencilAttachment;
        return {
            color: colorAttachment,
            depthStencil: depthStencilAttachment,
        };
    }

    private describePass(): GPURenderPassDescriptor {
        assert(this.attachments);
        return {
            label: "render-pass",
            colorAttachments: [this.attachments.color],
            depthStencilAttachment: this.attachments.depthStencil,
            timestampWrites: this.renderer.analytics.getRenderPassTimestamps(),
        } as GPURenderPassDescriptor;
    }

    private encodeBundle(): GPURenderBundle {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        const presentationFormat: GPUTextureFormat =
            this.renderer.presentationFormat;
        const encoder: GPURenderBundleEncoder =
            device.createRenderBundleEncoder({
                label: "render-bundle",
                colorFormats: [presentationFormat],
                depthStencilFormat: TextureFormats.depth,
            } as GPURenderBundleEncoderDescriptor);
        assert(this.pipeline);
        encoder.setPipeline(this.pipeline);
        encoder.setBindGroup(0, this.bindGroup);
        assert(this.buffer);
        encoder.drawIndirect(this.buffer, 0);
        return encoder.finish();
    }

    public synchronize(instances: int): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.data[1] = instances;
        assert(this.buffer);
        device.queue.writeBuffer(this.buffer, 0, this.data);
        //here only the instances part!
    }

    public render(encoder: GPUCommandEncoder): void {
        assert(this.passDescriptor);
        this.updateAttachment();
        const renderPass: GPURenderPassEncoder = encoder.beginRenderPass(
            this.passDescriptor,
        );
        assert(this.bundle);
        renderPass.executeBundles([this.bundle]);
        renderPass.end();
    }

    private updateAttachment(): void {
        assert(this.attachments);
        const view: GPUTextureView = this.renderer.requestContextView();
        this.attachments.color.view = view;
    }
}
