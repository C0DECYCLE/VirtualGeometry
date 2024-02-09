/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { assert } from "../utilities/utils.js";
import { EmptyCallback, Nullable, float } from "../utils.type.js";
import { Camera } from "./Camera.js";
import { Controller } from "./Controller.js";
import { PersistentThreadGroups, WebGPURequirements } from "../constants.js";
import { GeometryHandler } from "../handlers/GeometryHandler.js";
import { Analytics } from "./Analytics.js";
import { UniformHandler } from "../handlers/UniformHandler.js";
import { EntityHandler } from "../handlers/EntityHandler.js";
import { DrawHandler } from "../handlers/DrawHandler.js";
import { ShaderHandler } from "../handlers/ShaderHandler.js";
import { TextureHandler } from "../handlers/TextureHandler.js";
import { AttachmentHandler } from "../handlers/AttachmentHandler.js";
import { PipelineHandler } from "../handlers/PipelineHandler.js";
import { BindGroupHandler } from "../handlers/BindGroupHandler.js";
import { GeometryKey } from "../core.type.js";
import { Entity } from "./Entity.js";
import { ClusterHandler } from "../handlers/ClusterHandler.js";
import { InstanceHandler } from "../handlers/InstanceHandler.js";

export class Renderer {
    public readonly analytics: Analytics;
    public readonly handlers: {
        uniform: UniformHandler;
        geometry: GeometryHandler;
        entity: EntityHandler;
        shader: ShaderHandler;
        texture: TextureHandler;
        attachment: AttachmentHandler;
        pipeline: PipelineHandler;
        bindGroup: BindGroupHandler;
        instance: InstanceHandler;
        cluster: ClusterHandler;
        draw: DrawHandler;
    };
    private camera: Nullable<Camera>;
    private control: Nullable<Controller>;

    public readonly presentationFormat: GPUTextureFormat;
    public canvas: Nullable<HTMLCanvasElement>;
    private adapter: Nullable<GPUAdapter>;
    public device: Nullable<GPUDevice>;
    private context: Nullable<GPUCanvasContext>;

    private isPrepared: boolean;
    private isFrozen: boolean;

    public constructor() {
        assert(navigator.gpu);
        this.analytics = new Analytics();
        this.handlers = {
            uniform: new UniformHandler(),
            geometry: new GeometryHandler(),
            entity: new EntityHandler(this),
            shader: new ShaderHandler(),
            texture: new TextureHandler(),
            attachment: new AttachmentHandler(this),
            pipeline: new PipelineHandler(this),
            bindGroup: new BindGroupHandler(this),
            instance: new InstanceHandler(this),
            cluster: new ClusterHandler(this),
            draw: new DrawHandler(this),
        };
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.canvas = null;
        this.adapter = null;
        this.device = null;
        this.context = null;
        this.isPrepared = false;
        this.isFrozen = false;
    }

    public isReady(): boolean {
        return this.isPrepared;
    }

    public async import(key: GeometryKey, path: string): Promise<void> {
        await this.handlers.geometry.import(key, path);
    }

    public async prepare(): Promise<void> {
        assert(!this.isPrepared);
        await this.prepareGPU();
        assert(this.device);
        this.analytics.prepare(this.device);
        await this.prepareHandlers();
        this.prepareCameraControl();
        assert(this.canvas);
        this.isPrepared = true;
    }
    private async prepareGPU(): Promise<void> {
        this.canvas = this.createCanvas();
        this.adapter = await this.requestAdapter();
        this.device = await this.requestDevice(this.adapter);
        this.context = this.getContext(this.device);
    }

    private createCanvas(): HTMLCanvasElement {
        const canvas: HTMLCanvasElement = document.createElement("canvas");
        canvas.width = document.body.clientWidth * devicePixelRatio;
        canvas.height = document.body.clientHeight * devicePixelRatio;
        canvas.style.position = "absolute";
        canvas.style.top = "0px";
        canvas.style.left = "0px";
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        document.body.appendChild(canvas);
        return canvas;
    }

    private async requestAdapter(): Promise<GPUAdapter> {
        assert(navigator.gpu);
        const adapter: Nullable<GPUAdapter> =
            await navigator.gpu.requestAdapter();
        assert(adapter);
        return adapter;
    }

    private async requestDevice(adapter: GPUAdapter): Promise<GPUDevice> {
        const device: Nullable<GPUDevice> = await adapter.requestDevice({
            requiredFeatures: WebGPURequirements,
        } as GPUDeviceDescriptor);
        assert(device);
        return device;
    }

    private getContext(device: GPUDevice): GPUCanvasContext {
        assert(this.canvas);
        const context: Nullable<GPUCanvasContext> =
            this.canvas.getContext("webgpu");
        assert(context);
        context.configure({
            device: device,
            format: this.presentationFormat,
        } as GPUCanvasConfiguration);
        return context;
    }

    public requestContextView(): GPUTextureView {
        assert(this.context);
        return this.context.getCurrentTexture().createView();
    }

    private async prepareHandlers(): Promise<void> {
        assert(this.device && this.canvas);
        this.handlers.uniform.prepare(this.device);
        this.handlers.geometry.prepare(this.device);
        this.handlers.entity.prepare();
        await this.handlers.shader.prepare(this.device);
        this.handlers.texture.prepare(this.device, this.canvas);
        this.handlers.attachment.prepare();
        await this.handlers.pipeline.prepare();
        this.handlers.instance.prepare();
        this.handlers.cluster.prepare();
        this.handlers.draw.prepare();
        this.handlers.bindGroup.prepare();
        this.handlers.draw.encode();
    }

    private prepareCameraControl(): void {
        assert(this.canvas);
        this.camera = new Camera(this.canvas.width / this.canvas.height, 1000);
        this.camera.position.set(0, 0, 5);
        this.control = new Controller(this.canvas, this.camera);
    }

    public add(entity: Entity): void {
        entity.add(this.handlers.entity);
    }

    public run(update?: EmptyCallback): void {
        const tick = async (now: float): Promise<void> => {
            if (this.isPrepared) {
                this.frame(now, update);
            }
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    }

    private frame(now: float, update?: EmptyCallback): void {
        update?.();
        this.analytics.preFrame();
        this.gpuSync();
        this.execute();
        this.analytics.postFrame(now, this.handlers.draw);
    }

    private gpuSync(): void {
        assert(this.control && this.camera && this.device);
        this.control.update();
        this.handlers.uniform.viewProjection(this.camera.update());
        this.handlers.uniform.cameraPosition(this.camera.position);
        this.handlers.uniform.gpuSync(this.device);
        this.handlers.entity.gpuSync();
    }

    private execute(): void {
        assert(this.device);
        const encoder: GPUCommandEncoder = this.device.createCommandEncoder({
            label: "command-encoder",
        } as GPUObjectDescriptorBase);
        if (!this.isFrozen) {
            this.handlers.instance.setExecute(this.handlers.entity.count());
            this.handlers.instance.resetQueue();
            this.handlers.cluster.setExecute(PersistentThreadGroups);
            this.handlers.draw.setExecute(0);
            this.handlers.instance.execute(encoder);
            this.handlers.cluster.execute(encoder);
        }
        this.handlers.draw.execute(encoder);
        this.handlers.draw.resolve(encoder);
        this.analytics.resolve(encoder);
        this.device.queue.submit([encoder.finish()]);
    }

    public freeze(): void {
        this.isFrozen = true;
    }

    public unfreeze(): void {
        this.isFrozen = false;
    }
}
