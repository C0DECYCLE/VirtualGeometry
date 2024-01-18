/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, December 2023
 */

import { int, float } from "../types/utils.type.js";
import { Controller } from "./components/Controller.js";
import { Stats } from "./components/Stats.js";
import { RollingAverage } from "./utilities/RollingAverage.js";
import { createCanvas, loadOBJ, loadText, vec3Layout } from "./helper.js";
import { Camera } from "./components/Camera.js";
import { GPUTiming } from "./components/GPUTiming.js";
import { OBJParseResult } from "./components/OBJParser.js";
import { log } from "./utilities/logger.js";
import { dotit } from "./utilities/utils.js";

//////////// SETUP GPU ////////////

const canvas: HTMLCanvasElement = createCanvas();
const adapter: GPUAdapter = (await navigator.gpu?.requestAdapter())!;
const device: GPUDevice = (await adapter?.requestDevice({
    requiredFeatures: ["timestamp-query", "indirect-first-instance"],
} as GPUDeviceDescriptor))!;
const context: GPUCanvasContext = canvas.getContext("webgpu")!;
const presentationFormat: GPUTextureFormat =
    navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: presentationFormat,
} as GPUCanvasConfiguration);

//////////// CREATE CAMERA AND CONTROL ////////////

const camera: Camera = new Camera(canvas.width / canvas.height, 1000);
const control: Controller = new Controller(canvas, camera);

//////////// CREATE STATS ////////////

const stats: Stats = new Stats();
stats.set("frame delta", 0);
stats.set("gpu delta", 0);
stats.show();

const frameDelta: RollingAverage = new RollingAverage(60);
const cpuDelta: RollingAverage = new RollingAverage(60);
const gpuDelta: RollingAverage = new RollingAverage(60);

const gpuTiming: GPUTiming = new GPUTiming(device);

//////////// LOAD OBJ ////////////

const geometry: OBJParseResult = await loadOBJ("./resources/bunny.obj");

//////////// SETUP UNIFORM ////////////

const uniformLayout = 4 * 4;
const uniformData: Float32Array = new Float32Array(uniformLayout);
const uniformBuffer: GPUBuffer = device.createBuffer({
    label: "uniform buffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);

//////////// SETUP VERTICES ////////////

const vertexData: Float32Array = geometry.vertices;
const vertexBuffer: GPUBuffer = device.createBuffer({
    label: "vertex buffer",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(vertexBuffer, 0, vertexData);
log("vertices", dotit(vertexData.length / 8));

//////////// SETUP INDICES ////////////

const indexData: Uint32Array = geometry.indices!;
const indexBuffer: GPUBuffer = device.createBuffer({
    label: "index buffer",
    size: indexData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(indexBuffer, 0, indexData);
log("indices", dotit(indexData.length));
log("triangles", dotit(indexData.length / 3));

//////////// SETUP INSTANCES ////////////

const instanceLayout: int = vec3Layout;
const instanceCount: int = 1;
const instanceData: Float32Array = new Float32Array(
    instanceLayout * instanceCount,
);
const instanceBuffer: GPUBuffer = device.createBuffer({
    label: "instance buffer",
    size: instanceData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(instanceBuffer, 0, instanceData);
log("instances", dotit(instanceCount));

//////////// SETUP INDIRECTS ////////////

const indirectData: Uint32Array = new Uint32Array([
    geometry.indicesCount!,
    instanceCount,
    0,
    0,
]);
const indirectBuffer: GPUBuffer = device.createBuffer({
    label: "indirect buffer",
    size: indirectData.byteLength,
    usage:
        GPUBufferUsage.INDIRECT |
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(indirectBuffer, 0, indirectData);

//////////// LOAD SHADER ////////////

const renderShader: GPUShaderModule = device.createShaderModule({
    label: "render shader",
    code: await loadText("./shaders/render.wgsl"),
} as GPUShaderModuleDescriptor);

//////////// TEXTURES ////////////

const depthFormat: GPUTextureFormat = "depth24plus";
const depthTexture: GPUTexture = device.createTexture({
    label: "depth texture",
    size: [canvas.width, canvas.height],
    format: depthFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
} as GPUTextureDescriptor);

const textureViews = {
    depth: depthTexture.createView(),
};

//////////// CREATE PIPELINE ////////////

const pipeline: GPURenderPipeline = await device.createRenderPipelineAsync({
    label: "render pipeline",
    layout: "auto",
    vertex: {
        module: renderShader,
        entryPoint: "vs",
    } as GPUVertexState,
    fragment: {
        module: renderShader,
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
        format: depthFormat,
    } as GPUDepthStencilState,
} as GPURenderPipelineDescriptor);

//////////// CREATE BINDGROUP ////////////

const bindGroup: GPUBindGroup = device.createBindGroup({
    label: "render bindgroup",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: { buffer: uniformBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 1,
            resource: { buffer: indexBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 2,
            resource: { buffer: vertexBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 3,
            resource: { buffer: instanceBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
    ],
} as GPUBindGroupDescriptor);

//////////// SETUP RENDERPASS DESCRIPTORS ////////////

const colorAttachment: GPURenderPassColorAttachment = {
    label: "color attachment",
    view: context!.getCurrentTexture().createView(),
    clearValue: [0.3, 0.3, 0.3, 1],
    loadOp: "clear",
    storeOp: "store",
} as GPURenderPassColorAttachment;

const depthStencilAttachment: GPURenderPassDepthStencilAttachment = {
    label: "depth stencil attachment",
    view: textureViews.depth,
    depthClearValue: 1,
    depthLoadOp: "clear",
    depthStoreOp: "store",
} as GPURenderPassDepthStencilAttachment;

const renderPassDescriptor: GPURenderPassDescriptor = {
    label: "render pass",
    colorAttachments: [colorAttachment],
    depthStencilAttachment: depthStencilAttachment,
    timestampWrites: gpuTiming.timestampWrites,
} as GPURenderPassDescriptor;

//////////// CREATE BUNDLE ////////////

const bundleEncoder: GPURenderBundleEncoder = device.createRenderBundleEncoder({
    label: "render bundle",
    colorFormats: [presentationFormat],
    depthStencilFormat: depthFormat,
} as GPURenderBundleEncoderDescriptor);
bundleEncoder.setPipeline(pipeline);
bundleEncoder.setBindGroup(0, bindGroup);
bundleEncoder.drawIndirect(indirectBuffer, 0);
const bundle: GPURenderBundle = bundleEncoder.finish();

//////////// EACH FRAME ////////////

async function frame(now: float): Promise<void> {
    stats.time("cpu delta");

    //////////// UPDATE CONTROL CAMERA UNIFORMS ////////////

    control.update();
    camera.update().store(uniformData, 0);
    device!.queue.writeBuffer(uniformBuffer, 0, uniformData);

    //////////// RENDER FRAME ////////////

    colorAttachment.view = context!.getCurrentTexture().createView();

    const renderEncoder: GPUCommandEncoder = device!.createCommandEncoder({
        label: "render command encoder",
    } as GPUObjectDescriptorBase);

    const renderPass: GPURenderPassEncoder =
        renderEncoder.beginRenderPass(renderPassDescriptor);
    renderPass.executeBundles([bundle]);
    renderPass.end();

    gpuTiming.resolve(renderEncoder);

    const renderCommandBuffer: GPUCommandBuffer = renderEncoder.finish();
    device!.queue.submit([renderCommandBuffer]);

    //////////// UPDATE STATS ////////////

    stats.time("cpu delta", "cpu delta");
    cpuDelta.sample(stats.get("cpu delta")!);

    stats.set("frame delta", now - stats.get("frame delta")!);
    frameDelta.sample(stats.get("frame delta")!);

    gpuTiming.readback((ms: float) => {
        stats.set("gpu delta", ms);
        gpuDelta.sample(ms);
    });

    // prettier-ignore
    stats.update(`
        <b>frame rate: ${(1_000 / frameDelta.get()).toFixed(0)} fps</b><br>
        frame delta: ${frameDelta.get().toFixed(2)} ms<br>
        <br>
        <b>cpu rate: ${(1_000 / cpuDelta.get()).toFixed(0)} fps</b><br>
        cpu delta: ${cpuDelta.get().toFixed(2)} ms<br>
        <br>
        <b>gpu rate: ${(1_000 / gpuDelta.get()).toFixed(0)} fps</b><br>
        gpu delta: ${gpuDelta.get().toFixed(2)} ms<br>
    `);
    stats.set("frame delta", now);

    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
