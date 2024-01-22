/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { RollingAverage } from "../utilities/RollingAverage.js";
import { Vec3 } from "../utilities/Vec3.js";
import { log } from "../utilities/logger.js";
import { assert, clear, dotit } from "../utilities/utils.js";
import { EmptyCallback, Nullable, float, int } from "../utils.type.js";
import { Camera } from "./Camera.js";
import { Controller } from "./Controller.js";
import { GPUTiming } from "./GPUTiming.js";
import { OBJParseResult, OBJParser } from "./OBJParser.js";
import { Stats } from "./Stats.js";
import { vec3Layout } from "./constants.js";

export class Renderer {
    private static readonly Features: string[] = [
        "timestamp-query",
        "indirect-first-instance",
    ];

    private isInitialized: boolean = false;

    private geometry: OBJParseResult;

    private canvas: HTMLCanvasElement;
    private adapter: GPUAdapter;
    private device: GPUDevice;
    private context: GPUCanvasContext;

    private camera: Camera;
    private control: Controller;

    private static readonly AnalyticSamples: int = 60;
    private stats: Stats;
    private frameDelta: RollingAverage;
    private cpuDelta: RollingAverage;
    private gpuDelta: RollingAverage;
    private gpuTiming: GPUTiming;

    private uniformsData: Float32Array;
    private uniformsBuffer: GPUBuffer;

    private verticesData: Float32Array;
    private verticesBuffer: GPUBuffer;

    private indicesData: Uint32Array;
    private indicesBuffer: GPUBuffer;

    private instancesCount: int = 1;
    private instancesData: Float32Array;
    private instancesBuffer: GPUBuffer;

    private indirectData: Uint32Array;
    private indirectBuffer: GPUBuffer;

    private static readonly Shaders = {
        default: "./shaders/render.wgsl",
    };
    private defaultShader: GPUShaderModule;

    private static readonly TextureFormats = {
        depth: "depth24plus",
    };
    private readonly presentationFormat: GPUTextureFormat;
    private textures: { depth: GPUTexture };
    private textureViews: { depth: GPUTextureView };

    private pipeline: GPURenderPipeline;

    private bindGroup: GPUBindGroup;

    private attachments: {
        color: GPURenderPassColorAttachment;
        depthStencil: GPURenderPassDepthStencilAttachment;
    };

    private renderPassDescriptor: GPURenderPassDescriptor;

    private bundle: GPURenderBundle;

    public constructor() {
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    }

    public async importGeometry(key: string, path: string): Promise<void> {
        assert(!this.isInitialized);
        const geometry: OBJParseResult = OBJParser.Standard.parse(
            await this.loadText(path),
            true,
        );
        //build tree
        const triangleIndices: [int, int, int][] = [];
        const triangleIdQueue: int[] = [];
        const triangleIdToClusterId: int[] = [];
        for (let i: int = 0; i < geometry.indicesCount! / 3; i++) {
            triangleIndices.push([
                geometry.indices![i * 3 + 0],
                geometry.indices![i * 3 + 1],
                geometry.indices![i * 3 + 2],
            ]);
            triangleIdQueue.push(i);
            triangleIdToClusterId.push(0);
        }

        type BoundingSphere = {
            sum: Vec3;
            n: int;
            center: Vec3;
            radius: float;
        };
        function getVerices(indices: [int, int, int]): {
            a: Vec3;
            b: Vec3;
            c: Vec3;
        } {
            const a: Vec3 = new Vec3(
                geometry.vertices[indices[0] * 4 + 0],
                geometry.vertices[indices[0] * 4 + 1],
                geometry.vertices[indices[0] * 4 + 2],
            );
            const b: Vec3 = new Vec3(
                geometry.vertices[indices[1] * 4 + 0],
                geometry.vertices[indices[1] * 4 + 1],
                geometry.vertices[indices[1] * 4 + 2],
            );
            const c: Vec3 = new Vec3(
                geometry.vertices[indices[2] * 4 + 0],
                geometry.vertices[indices[2] * 4 + 1],
                geometry.vertices[indices[2] * 4 + 2],
            );
            return { a, b, c };
        }
        function boundingSphere(triangleIds: int[]): BoundingSphere {
            const sum: Vec3 = new Vec3();
            for (let i: int = 0; i < triangleIds.length; i++) {
                const { a, b, c } = getVerices(triangleIndices[triangleIds[i]]);
                sum.add(a);
                sum.add(b);
                sum.add(c);
            }
            const n: int = triangleIds.length * 3;
            const center = sum.clone().scale(1 / n);
            let radius: float = 0;
            for (let i: int = 0; i < triangleIds.length; i++) {
                const { a, b, c } = getVerices(triangleIndices[triangleIds[i]]);
                radius = Math.max(
                    radius,
                    a.sub(center).lengthQuadratic(),
                    b.sub(center).lengthQuadratic(),
                    c.sub(center).lengthQuadratic(),
                );
            }
            return { sum: sum, n: n, center: center, radius: radius };
        }
        function boundingSphereRunning(
            base: BoundingSphere,
            triangleIds: int[],
            additional: int,
        ): BoundingSphere {
            const { a, b, c } = getVerices(triangleIndices[additional]);
            const sum: Vec3 = base.sum.clone();
            sum.add(a);
            sum.add(b);
            sum.add(c);
            const n: int = base.n + 1;
            const center = sum.clone().scale(1 / n);
            let radius: float = 0;
            for (let i: int = 0; i < triangleIds.length + 1; i++) {
                if (i >= triangleIds.length && additional === undefined) {
                    break;
                }
                let indices: [int, int, int] =
                    i >= triangleIds.length
                        ? triangleIndices[additional!]
                        : triangleIndices[triangleIds[i]];

                const { a, b, c } = getVerices(indices);
                radius = Math.max(
                    radius,
                    a.sub(center).lengthQuadratic(),
                    b.sub(center).lengthQuadratic(),
                    c.sub(center).lengthQuadratic(),
                );
            }
            return { sum: sum, n: n, center: center, radius: radius };
        }

        const pre: float = performance.now();

        let clusterId = 0;
        while (triangleIdQueue.length !== 0) {
            const first: int = triangleIdQueue.pop()!;
            triangleIdToClusterId[first] = clusterId;

            const inCluster: int[] = [first];
            let clusterBounding: BoundingSphere = boundingSphere(inCluster);
            while (inCluster.length < 128 && triangleIdQueue.length !== 0) {
                let bestI: int = -1;
                let bestBounding: BoundingSphere;
                let bestIncrease: float = Infinity;
                for (let i: int = 0; i < triangleIdQueue.length; i++) {
                    const possibleId: int = triangleIdQueue[i];
                    const possibleBounding: BoundingSphere =
                        boundingSphereRunning(
                            clusterBounding,
                            inCluster,
                            possibleId,
                        );
                    const possibleIncrease: float =
                        possibleBounding.radius - clusterBounding.radius;
                    if (possibleIncrease < bestIncrease) {
                        bestI = i;
                        bestBounding = possibleBounding;
                        bestIncrease = possibleIncrease;
                    }
                }
                assert(bestI !== -1);
                assert(bestBounding!);
                const bestId: int = triangleIdQueue[bestI];
                triangleIdQueue[bestI] =
                    triangleIdQueue[triangleIdQueue.length - 1];
                triangleIdQueue.pop();
                triangleIdToClusterId[bestId] = clusterId;
                inCluster.push(bestId);
                clusterBounding = bestBounding;
            }

            if (clusterId % 10 === 0) {
                log(clusterId, inCluster.length);
            }
            clusterId++;
        }

        log(dotit(clusterId), "clusters", dotit(performance.now() - pre), "ms");

        let indices: Uint32Array = new Uint32Array(geometry.indicesCount! * 3);
        for (let i: int = 0; i < triangleIndices.length; i++) {
            indices[i * 9 + 0] = geometry.indices![i * 3 + 0];
            indices[i * 9 + 1] = i;
            indices[i * 9 + 2] = triangleIdToClusterId[i];
            indices[i * 9 + 3] = geometry.indices![i * 3 + 1];
            indices[i * 9 + 4] = i;
            indices[i * 9 + 5] = triangleIdToClusterId[i];
            indices[i * 9 + 6] = geometry.indices![i * 3 + 2];
            indices[i * 9 + 7] = i;
            indices[i * 9 + 8] = triangleIdToClusterId[i];
        }
        geometry.indices = indices;
        this.geometry = geometry;
    }

    private async loadText(path: string): Promise<string> {
        return await fetch(path).then(
            async (response: Response) => await response.text(),
        );
    }

    public async initialize(): Promise<void> {
        this.isInitialized = true;
        await this.initializeGPU();
        this.initializeAnalytics();
        this.initializeCameraControl();
        this.initializeUniforms();
        this.initializeVertices();
        this.initializeIndices();
        this.initializeInstances();
        this.initializeIndirect();
        await this.initializeShaders();
        this.initializeTextures();
        await this.initializePipeline();
        this.initializeBindGroup();
        this.initializeAttachments();
        this.initializePassDescriptors();
        this.initializeBundle();
    }

    private async initializeGPU(): Promise<void> {
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
            requiredFeatures: Renderer.Features,
        } as GPUDeviceDescriptor);
        assert(device);
        return device;
    }

    private getContext(device: GPUDevice): GPUCanvasContext {
        const context: Nullable<GPUCanvasContext> =
            this.canvas.getContext("webgpu");
        assert(context);
        context.configure({
            device: device,
            format: this.presentationFormat,
        } as GPUCanvasConfiguration);
        return context;
    }

    private initializeAnalytics(): void {
        this.stats = this.createStats();
        this.frameDelta = new RollingAverage(Renderer.AnalyticSamples);
        this.cpuDelta = new RollingAverage(Renderer.AnalyticSamples);
        this.gpuDelta = new RollingAverage(Renderer.AnalyticSamples);
        this.gpuTiming = new GPUTiming(this.device);
    }

    private createStats(): Stats {
        const stats: Stats = new Stats();
        stats.set("frame delta", 0);
        stats.set("gpu delta", 0);
        stats.show();
        return stats;
    }

    private initializeCameraControl(): void {
        this.camera = new Camera(this.canvas.width / this.canvas.height, 1000);
        this.camera.position.set(0, 0, 5);
        this.control = new Controller(this.canvas, this.camera);
    }

    private initializeUniforms(): void {
        const uniformsLayout = 4 * 4;
        this.uniformsData = new Float32Array(uniformsLayout);
        this.uniformsBuffer = this.device.createBuffer({
            label: "main-uniforms-buffer",
            size: this.uniformsData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
    }

    private initializeVertices(): void {
        this.verticesData = this.geometry.vertices;
        this.verticesBuffer = this.device.createBuffer({
            label: "main-vertices-buffer",
            size: this.verticesData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        this.device.queue.writeBuffer(
            this.verticesBuffer,
            0,
            this.verticesData,
        );
        log("main-vertices", dotit(this.geometry.verticesCount));
    }

    private initializeIndices(): void {
        assert(this.geometry.indices);
        this.indicesData = this.geometry.indices;
        this.indicesBuffer = this.device.createBuffer({
            label: "main-indices-buffer",
            size: this.indicesData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        this.device.queue.writeBuffer(this.indicesBuffer, 0, this.indicesData);
        assert(this.geometry.indicesCount);
        log("main-indices", dotit(this.geometry.indicesCount));
        log("main-triangles", dotit(this.geometry.indicesCount / 3));
    }

    private initializeInstances(): void {
        const instanceLayout: int = vec3Layout;
        this.instancesData = new Float32Array(
            instanceLayout * this.instancesCount,
        );
        this.instancesBuffer = this.device.createBuffer({
            label: "main-instances-buffer",
            size: this.instancesData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        this.device.queue.writeBuffer(
            this.instancesBuffer,
            0,
            this.instancesData,
        );
        log("main-instances", dotit(this.instancesCount));
    }

    private initializeIndirect(): void {
        assert(this.geometry.indicesCount);
        this.indirectData = new Uint32Array([
            this.geometry.indicesCount,
            this.instancesCount,
            0,
            0,
        ]);
        this.indirectBuffer = this.device.createBuffer({
            label: "main-indirect-buffer",
            size: this.indirectData.byteLength,
            usage:
                GPUBufferUsage.INDIRECT |
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        this.device.queue.writeBuffer(
            this.indirectBuffer,
            0,
            this.indirectData,
        );
    }

    private async initializeShaders(): Promise<void> {
        this.defaultShader = await this.requestShader(Renderer.Shaders.default);
    }

    private async requestShader(path: string): Promise<GPUShaderModule> {
        return this.device.createShaderModule({
            label: `shader-module-${path}`,
            code: await this.loadText(path),
        } as GPUShaderModuleDescriptor);
    }

    private initializeTextures(): void {
        const depthTexture: GPUTexture = this.device.createTexture({
            label: "depth texture",
            size: [this.canvas.width, this.canvas.height],
            format: Renderer.TextureFormats.depth,
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING,
        } as GPUTextureDescriptor);
        this.textures = {
            depth: depthTexture,
        };
        this.textureViews = {
            depth: this.textures.depth.createView(),
        };
    }

    private async initializePipeline(): Promise<void> {
        this.pipeline = await this.device.createRenderPipelineAsync({
            label: "main-render-pipeline",
            layout: "auto",
            vertex: {
                module: this.defaultShader,
                entryPoint: "vs",
            } as GPUVertexState,
            fragment: {
                module: this.defaultShader,
                entryPoint: "fs",
                targets: [{ format: this.presentationFormat }],
            } as GPUFragmentState,
            primitive: {
                topology: "triangle-list",
                cullMode: "none", //"back"
            } as GPUPrimitiveState,
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: Renderer.TextureFormats.depth,
            } as GPUDepthStencilState,
        } as GPURenderPipelineDescriptor);
    }

    private initializeBindGroup(): void {
        this.bindGroup = this.device.createBindGroup({
            label: "render bindgroup",
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.uniformsBuffer,
                    } as GPUBindingResource,
                } as GPUBindGroupEntry,
                {
                    binding: 1,
                    resource: {
                        buffer: this.indicesBuffer,
                    } as GPUBindingResource,
                } as GPUBindGroupEntry,
                {
                    binding: 2,
                    resource: {
                        buffer: this.verticesBuffer,
                    } as GPUBindingResource,
                } as GPUBindGroupEntry,
                {
                    binding: 3,
                    resource: {
                        buffer: this.instancesBuffer,
                    } as GPUBindingResource,
                } as GPUBindGroupEntry,
            ],
        } as GPUBindGroupDescriptor);
    }

    private initializeAttachments(): void {
        const colorAttachment: GPURenderPassColorAttachment = {
            label: "color attachment",
            view: this.context.getCurrentTexture().createView(),
            clearValue: [0.3, 0.3, 0.3, 1],
            loadOp: "clear",
            storeOp: "store",
        } as GPURenderPassColorAttachment;
        const depthStencilAttachment: GPURenderPassDepthStencilAttachment = {
            label: "depth stencil attachment",
            view: this.textureViews.depth,
            depthClearValue: 1,
            depthLoadOp: "clear",
            depthStoreOp: "store",
        } as GPURenderPassDepthStencilAttachment;
        this.attachments = {
            color: colorAttachment,
            depthStencil: depthStencilAttachment,
        };
    }

    private initializePassDescriptors(): void {
        this.renderPassDescriptor = {
            label: "main-render-pass",
            colorAttachments: [this.attachments.color],
            depthStencilAttachment: this.attachments.depthStencil,
            timestampWrites: this.gpuTiming.timestampWrites,
        } as GPURenderPassDescriptor;
    }

    private initializeBundle(): void {
        const bundleEncoder: GPURenderBundleEncoder =
            this.device.createRenderBundleEncoder({
                label: "render bundle",
                colorFormats: [this.presentationFormat],
                depthStencilFormat: Renderer.TextureFormats.depth,
            } as GPURenderBundleEncoderDescriptor);
        bundleEncoder.setPipeline(this.pipeline);
        bundleEncoder.setBindGroup(0, this.bindGroup);
        bundleEncoder.drawIndirect(this.indirectBuffer, 0);
        this.bundle = bundleEncoder.finish();
    }

    public run(update?: EmptyCallback): void {
        const tick = async (now: float): Promise<void> => {
            if (this.isInitialized) {
                update?.();
                this.frame(now);
            }
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    }

    private frame(now: float): void {
        this.preFrameAnalytics();
        this.updateInternals();
        this.updateAttachments();
        this.encodeCommands();
        this.postFrameAnalytics(now);
    }

    private preFrameAnalytics(): void {
        this.stats.time("cpu delta");
    }

    private updateInternals(): void {
        this.control.update();
        this.camera.update().store(this.uniformsData, 0);
        this.device.queue.writeBuffer(
            this.uniformsBuffer,
            0,
            this.uniformsData,
        );
    }

    private updateAttachments(): void {
        this.attachments.color.view = this.context
            .getCurrentTexture()
            .createView();
    }

    private encodeCommands(): void {
        const commandEncoder: GPUCommandEncoder =
            this.device.createCommandEncoder({
                label: "main-command-encoder",
            } as GPUObjectDescriptorBase);
        const renderPass: GPURenderPassEncoder = commandEncoder.beginRenderPass(
            this.renderPassDescriptor,
        );
        renderPass.executeBundles([this.bundle]);
        renderPass.end();
        this.gpuTiming.resolve(commandEncoder);
        const commandBuffer: GPUCommandBuffer = commandEncoder.finish();
        this.device.queue.submit([commandBuffer]);
    }

    private postFrameAnalytics(now: float): void {
        this.stats.time("cpu delta", "cpu delta");
        this.cpuDelta.sample(this.stats.get("cpu delta")!);
        this.stats.set("frame delta", now - this.stats.get("frame delta")!);
        this.frameDelta.sample(this.stats.get("frame delta")!);
        this.gpuTiming.readback((ms: float) => {
            this.stats.set("gpu delta", ms);
            this.gpuDelta.sample(ms);
        });
        // prettier-ignore
        this.stats.update(`
            <b>frame rate: ${(1_000 / this.frameDelta.get()).toFixed(
                0,
            )} fps</b><br>
            frame delta: ${this.frameDelta.get().toFixed(2)} ms<br>
            <br>
            <b>cpu rate: ${(1_000 / this.cpuDelta.get()).toFixed(0)} fps</b><br>
            cpu delta: ${this.cpuDelta.get().toFixed(2)} ms<br>
            <br>
            <b>gpu rate: ${(1_000 / this.gpuDelta.get()).toFixed(0)} fps</b><br>
            gpu delta: ${this.gpuDelta.get().toFixed(2)} ms<br>
        `);
        this.stats.set("frame delta", now);
    }
}
