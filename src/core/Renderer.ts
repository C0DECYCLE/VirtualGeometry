/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { assert } from "../utilities/utils.js";
import { EmptyCallback, Nullable, float } from "../utils.type.js";
import { Camera } from "../components/Camera.js";
import { Controller } from "../components/Controller.js";
import { WebGPURequirements } from "../constants.js";
import { GeometryHandler } from "./GeometryHandler.js";
import { Analytics } from "../components/Analytics.js";
import { UniformsHandler } from "./UniformsHandler.js";
import { EntityHandler } from "./EntityHandler.js";
import { DrawHandler } from "./DrawHandler.js";
import { ShaderHandler } from "./ShaderHandler.js";

export class Renderer {
    public readonly analytics: Analytics;
    public readonly uniformsHandler: UniformsHandler;
    public readonly geometryHandler: GeometryHandler;
    public readonly entityHandler: EntityHandler;
    public readonly shaderHandler: ShaderHandler;
    public readonly drawHandler: DrawHandler;
    private camera: Nullable<Camera>;
    private control: Nullable<Controller>;

    public readonly presentationFormat: GPUTextureFormat;
    public canvas: Nullable<HTMLCanvasElement>;
    private adapter: Nullable<GPUAdapter>;
    public device: Nullable<GPUDevice>;
    private context: Nullable<GPUCanvasContext>;

    private isPrepared: boolean;

    public constructor() {
        this.analytics = new Analytics();
        this.uniformsHandler = new UniformsHandler();
        this.geometryHandler = new GeometryHandler();
        this.entityHandler = new EntityHandler(this);
        this.shaderHandler = new ShaderHandler();
        this.drawHandler = new DrawHandler(this);
        assert(navigator.gpu);
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.canvas = null;
        this.adapter = null;
        this.device = null;
        this.context = null;
        this.isPrepared = false;
    }

    public isReady(): boolean {
        return this.isPrepared;
    }

    public async prepare(): Promise<void> {
        assert(!this.isPrepared);
        await this.prepareGPU();
        assert(this.device);
        this.analytics.prepare(this.device);
        this.uniformsHandler.prepare(this.device);
        this.geometryHandler.prepare(this.device);
        this.entityHandler.prepare();
        await this.shaderHandler.prepare(this.device);
        await this.drawHandler.prepare();
        this.prepareCameraControl();
        this.isPrepared = true;
        /////
        this.drawHandler.synchronize(this.geometryHandler.count.clusters);
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

    private prepareCameraControl(): void {
        assert(this.canvas);
        this.camera = new Camera(this.canvas.width / this.canvas.height, 1000);
        this.camera.position.set(0, 0, 5);
        this.control = new Controller(this.canvas, this.camera);
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
        this.synchronize();
        this.render();
        this.analytics.postFrame(now);
    }

    private synchronize(): void {
        assert(this.control && this.camera);
        this.control.update();
        this.camera.update().store(this.uniformsHandler.data, 0);
        assert(this.device);
        this.uniformsHandler.synchronize(this.device);
    }

    private render(): void {
        assert(this.device);
        const encoder: GPUCommandEncoder = this.device.createCommandEncoder({
            label: "command-encoder",
        } as GPUObjectDescriptorBase);
        this.drawHandler.render(encoder);
        this.analytics.resolve(encoder);
        this.device.queue.submit([encoder.finish()]);
    }
}
