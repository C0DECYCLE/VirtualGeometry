/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, December 2023
 */

import { int, float } from "../../types/utilities/utils.type.js";
import { Controller } from "../Controller.js";
import { Stats } from "../Stats.js";
import { RollingAverage } from "../RollingAverage.js";
import { createCanvas, loadOBJ, loadText } from "./helper.js";
import { Camera } from "./components/Camera.js";
import { GPUTiming } from "./components/GPUTiming.js";
import { OBJParseResult } from "../OBJParser.js";
import { log } from "../utilities/logger.js";
import { Vec3 } from "../utilities/Vec3.js";
import { Mat4 } from "../utilities/Mat4.js";
import { dotit } from "../utilities/utils.js";

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

//////////// CREATE LIGHT AND SHADOW ////////////

const shadowSize: int = 1024;
const shadowBias: int = 0.005;
const shadowRadius: float = 30;
const lightDirection: Vec3 = new Vec3(0.27, -0.71, -0.35).normalize().scale(-1);
const lightViewProjection: Mat4 = new Mat4().multiply(
    Mat4.View(new Vec3(0, 0, 0), lightDirection, new Vec3(0, 1, 0)),
    Mat4.Orthogonal(
        -shadowRadius,
        shadowRadius,
        shadowRadius,
        -shadowRadius,
        -shadowRadius,
        shadowRadius,
    ),
);

//////////// CREATE STATS ////////////

const stats: Stats = new Stats();
stats.set("frame delta", 0);
stats.set("gpu delta", 0);
stats.show();

const frameDelta: RollingAverage = new RollingAverage(60);
const cpuDelta: RollingAverage = new RollingAverage(60);
const gpuShadowDelta: RollingAverage = new RollingAverage(60);
const gpuTargetDelta: RollingAverage = new RollingAverage(60);
const gpuDeferredDelta: RollingAverage = new RollingAverage(60);

const shadowGPUTiming: GPUTiming = new GPUTiming(device);
const targetGPUTiming: GPUTiming = new GPUTiming(device);
const deferredGPUTiming: GPUTiming = new GPUTiming(device);

//////////// LOAD OBJ ////////////

const c: Vec3 = new Vec3(1, 1, 1);

const cube: OBJParseResult = await loadOBJ("./resources/cube.obj", c);
const icosphere: OBJParseResult = await loadOBJ("./resources/icosphere.obj", c);
const torus: OBJParseResult = await loadOBJ("./resources/torus.obj", c);
const cylinder: OBJParseResult = await loadOBJ("./resources/cylinder.obj", c);
const cone: OBJParseResult = await loadOBJ("./resources/cone.obj", c);
const suzanne: OBJParseResult = await loadOBJ("./resources/suzanne.obj", c);
const building: OBJParseResult = await loadOBJ("./resources/building.obj");

const geometries: OBJParseResult[] = [
    cube,
    icosphere,
    torus,
    cylinder,
    cone,
    suzanne,
    building,
];

//////////// SETUP UNIFORM ////////////

const vec3Layout: int = 3 + 1;
const uniformLayout = 4 * 4 + vec3Layout + 4 * 4 + (1 + 1 + 2);
const uniformData: Float32Array = new Float32Array(uniformLayout);
const uniformBuffer: GPUBuffer = device.createBuffer({
    label: "uniform buffer",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
lightDirection.store(uniformData, 16);
lightViewProjection.store(uniformData, 20);
uniformData.set([shadowSize, shadowBias], 36);
device!.queue.writeBuffer(uniformBuffer, 0, uniformData);

//////////// SETUP VERTICES ////////////

const vertexData: Float32Array = new Float32Array(
    geometries.flatMap((geometry: OBJParseResult) => [...geometry.vertices]),
);
const vertexBuffer: GPUBuffer = device.createBuffer({
    label: "vertex buffer",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(vertexBuffer, 0, vertexData);
log("vertices", dotit(vertexData.length / 8));

//////////// SETUP INDICES ////////////

const indexData: Uint32Array = new Uint32Array(
    geometries.flatMap((geometry: OBJParseResult) => [...geometry.indices!]),
);
const indexBuffer: GPUBuffer = device.createBuffer({
    label: "index buffer",
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(indexBuffer, 0, indexData);
log("indices", dotit(indexData.length));

//////////// SETUP PROBES ////////////

const probeCountX: int = 5;
const probeCountY: int = 3;
const probeCountZ: int = 5;
const probeCount: int = probeCountX * probeCountY * probeCountZ;
const probeLayout: int = vec3Layout + vec3Layout;
const probeData: Float32Array = new Float32Array(probeCount * probeLayout);

const spread: float = 16;
let i: int = 0;
for (let y: int = 0; y < probeCountY; y++) {
    for (let z: int = 0; z < probeCountZ; z++) {
        for (let x: int = 0; x < probeCountX; x++) {
            const position: Vec3 = new Vec3(
                x - Math.floor(probeCountX / 2),
                y - 0,
                z - Math.floor(probeCountZ / 2),
            ).scale(spread);

            const color: Vec3 = new Vec3();
            if (position.length() === 0) {
                color.set(0.7, 0.2, 0.3);
            } else if (y === 0) {
                color.set(0.9, 0.85, 0.95);
            } else {
                color.set(0.6, 0.7, 0.7);
            }
            color.set(Math.random(), Math.random(), Math.random());

            position.store(probeData, i * probeLayout + 0 * vec3Layout);
            color.store(probeData, i * probeLayout + 1 * vec3Layout);
            i++;
        }
    }
}

const probeBuffer: GPUBuffer = device.createBuffer({
    label: "probe buffer",
    size: probeData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(probeBuffer, 0, probeData);
log("probes", dotit(probeCount));

//////////// SETUP INSTANCES ////////////

type Entity = {
    position: Vec3;
    scaling: Vec3;
    color: Vec3;
    geometryId: int;
};

const entities: Entity[] = [
    {
        position: new Vec3(0, -0.5, 0),
        scaling: new Vec3(100, 1, 100),
        color: new Vec3(0.9, 0.85, 0.95),
        geometryId: 0,
    } as Entity,
    {
        position: new Vec3(0, 0.9, 0),
        scaling: new Vec3(1, 1, 1),
        color: new Vec3(0.7, 0.2, 0.3),
        geometryId: 5,
    } as Entity,
    {
        position: new Vec3(-14, 0, -4),
        scaling: new Vec3(2, 2, 2),
        color: new Vec3(1, 1, 1),
        geometryId: 6,
    } as Entity,
];

const debugProbes: boolean = true;
if (debugProbes) {
    for (let i: int = 0; i < probeCount; i++) {
        const position: Vec3 = new Vec3(
            probeData[i * probeLayout + 0 * vec3Layout + 0],
            probeData[i * probeLayout + 0 * vec3Layout + 1],
            probeData[i * probeLayout + 0 * vec3Layout + 2],
        );
        const color: Vec3 = new Vec3(
            probeData[i * probeLayout + 1 * vec3Layout + 0],
            probeData[i * probeLayout + 1 * vec3Layout + 1],
            probeData[i * probeLayout + 1 * vec3Layout + 2],
        );
        entities.push({
            position: position,
            scaling: new Vec3(0.5, 0.5, 0.5),
            color: color,
            geometryId: 1,
        } as Entity);
    }
}

const geometryCounts: int[] = new Array(geometries.length).fill(0);
const entityCount: int = entities.length;
const instanceLayout: int = vec3Layout + vec3Layout + vec3Layout;
const instanceData: Float32Array = new Float32Array(
    instanceLayout * entityCount,
);

entities.sort((a: Entity, b: Entity) => a.geometryId - b.geometryId);
entities.forEach((entity: Entity, i: int) => {
    entity.position.store(instanceData, i * instanceLayout + 0 * vec3Layout);
    entity.scaling.store(instanceData, i * instanceLayout + 1 * vec3Layout);
    entity.color.store(instanceData, i * instanceLayout + 2 * vec3Layout);
    geometryCounts[entity.geometryId]++;
});

const instanceBuffer: GPUBuffer = device.createBuffer({
    label: "instance buffer",
    size: instanceData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(instanceBuffer, 0, instanceData);
log("instances/entities", dotit(entityCount));

//////////// SETUP INDIRECTS ////////////

const indirectData: Uint32Array = new Uint32Array(5 * geometries.length);
let totalIndices: int = 0;
let totalPositions: int = 0;
let totalN: int = 0;

geometries.forEach((geometry: OBJParseResult, i: int) => {
    indirectData.set(
        [
            geometry.indicesCount!,
            geometryCounts[i],
            totalIndices,
            totalPositions,
            totalN,
        ],
        5 * i,
    );
    totalIndices += geometry.indicesCount!;
    totalPositions += geometry.verticesCount;
    totalN += geometryCounts[i];
});

const indirectBuffer: GPUBuffer = device.createBuffer({
    label: "indirect buffer",
    size: indirectData.byteLength,
    usage:
        GPUBufferUsage.INDIRECT |
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
} as GPUBufferDescriptor);
device.queue.writeBuffer(indirectBuffer, 0, indirectData);

//////////// LOAD SHADER ////////////

const shadowShader: GPUShaderModule = device.createShaderModule({
    label: "shadow shader",
    code: await loadText("./shaders/globalillumination/shadow.wgsl"),
} as GPUShaderModuleDescriptor);

const targetShader: GPUShaderModule = device.createShaderModule({
    label: "target shader",
    code: await loadText("./shaders/globalillumination/targets.wgsl"),
} as GPUShaderModuleDescriptor);

const deferredShader: GPUShaderModule = device.createShaderModule({
    label: "deferred shader",
    code: await loadText("./shaders/globalillumination/deferred.wgsl"),
} as GPUShaderModuleDescriptor);

//////////// RENDER TARGETS ////////////

const shadowFormat: GPUTextureFormat = "depth32float";
const shadowTarget: GPUTexture = device.createTexture({
    label: "shadow target texture",
    size: [shadowSize, shadowSize, 1],
    format: shadowFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
} as GPUTextureDescriptor);
const shadowSampler: GPUSampler = device.createSampler({
    label: "shadow sampler",
    compare: "less",
} as GPUSamplerDescriptor);

const colorFormat: GPUTextureFormat = "bgra8unorm";
const colorTarget: GPUTexture = device.createTexture({
    label: "color target texture",
    size: [canvas.width, canvas.height],
    format: colorFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

const depthFormat: GPUTextureFormat = "depth24plus";
const depthTarget: GPUTexture = device.createTexture({
    label: "depth target texture",
    size: [canvas.width, canvas.height],
    format: depthFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
} as GPUTextureDescriptor);

const normalFormat: GPUTextureFormat = "rgba16float";
const normalTarget: GPUTexture = device.createTexture({
    label: "normal target texture",
    size: [canvas.width, canvas.height],
    format: normalFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

const positionFormat: GPUTextureFormat = "rgba32float";
const positionTarget: GPUTexture = device.createTexture({
    label: "position target texture",
    size: [canvas.width, canvas.height],
    format: positionFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});

const targetViews = {
    shadow: shadowTarget.createView(),
    color: colorTarget.createView(),
    depth: depthTarget.createView(),
    normal: normalTarget.createView(),
    position: positionTarget.createView(),
};

//////////// CREATE PIPELINE ////////////

const shadowPipeline: GPURenderPipeline =
    await device.createRenderPipelineAsync({
        label: "shadow pipeline",
        layout: "auto",
        vertex: {
            module: shadowShader,
            entryPoint: "vs",
        } as GPUVertexState,
        primitive: {
            topology: "triangle-list",
            cullMode: "back",
        } as GPUPrimitiveState,
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: "less",
            format: shadowFormat,
        } as GPUDepthStencilState,
    } as GPURenderPipelineDescriptor);

const targetPipeline: GPURenderPipeline =
    await device.createRenderPipelineAsync({
        label: "target pipeline",
        layout: "auto",
        vertex: {
            module: targetShader,
            entryPoint: "vs",
        } as GPUVertexState,
        fragment: {
            module: targetShader,
            entryPoint: "fs",
            targets: [
                { format: colorFormat },
                { format: normalFormat },
                { format: positionFormat },
            ],
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

const deferredPipeline: GPURenderPipeline =
    await device.createRenderPipelineAsync({
        label: "deferred pipeline",
        layout: "auto",
        vertex: {
            module: deferredShader,
            entryPoint: "vs",
        } as GPUVertexState,
        fragment: {
            module: deferredShader,
            entryPoint: "fs",
            targets: [{ format: presentationFormat }],
        } as GPUFragmentState,
        primitive: {
            topology: "triangle-list",
            cullMode: "back",
        } as GPUPrimitiveState,
    } as GPURenderPipelineDescriptor);

//////////// CREATE BINDGROUP ////////////

const shadowBindGroup: GPUBindGroup = device.createBindGroup({
    label: "shadow bindgroup",
    layout: shadowPipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: { buffer: uniformBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 1,
            resource: { buffer: vertexBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 2,
            resource: { buffer: instanceBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
    ],
} as GPUBindGroupDescriptor);

const targetBindGroup: GPUBindGroup = device.createBindGroup({
    label: "target bindgroup",
    layout: targetPipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: { buffer: uniformBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 1,
            resource: { buffer: vertexBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        {
            binding: 2,
            resource: { buffer: instanceBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
    ],
} as GPUBindGroupDescriptor);

const deferredBindGroup: GPUBindGroup = device.createBindGroup({
    label: "deferred bindgroup",
    layout: deferredPipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: { buffer: uniformBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
        { binding: 1, resource: targetViews.color } as GPUBindGroupEntry,
        { binding: 2, resource: targetViews.depth } as GPUBindGroupEntry,
        { binding: 3, resource: targetViews.normal } as GPUBindGroupEntry,
        { binding: 4, resource: targetViews.position } as GPUBindGroupEntry,
        { binding: 5, resource: targetViews.shadow } as GPUBindGroupEntry,
        { binding: 6, resource: shadowSampler } as GPUBindGroupEntry,
        {
            binding: 7,
            resource: { buffer: probeBuffer } as GPUBindingResource,
        } as GPUBindGroupEntry,
    ],
} as GPUBindGroupDescriptor);

//////////// SETUP RENDERPASS DESCRIPTORS ////////////

const shadowDepthStencilAttachment: GPURenderPassDepthStencilAttachment = {
    label: "shadow depth stencil attachment",
    view: targetViews.shadow,
    depthClearValue: 1,
    depthLoadOp: "clear",
    depthStoreOp: "store",
} as GPURenderPassDepthStencilAttachment;

const shadowPassDescriptor: GPURenderPassDescriptor = {
    label: "shadow render pass",
    colorAttachments: [],
    depthStencilAttachment: shadowDepthStencilAttachment,
    timestampWrites: shadowGPUTiming.timestampWrites,
} as GPURenderPassDescriptor;

const colorTargetAttachment: GPURenderPassColorAttachment = {
    label: "color target attachment",
    view: targetViews.color,
    clearValue: [0, 0, 0, 1],
    loadOp: "clear",
    storeOp: "store",
} as GPURenderPassColorAttachment;

const depthStencilAttachment: GPURenderPassDepthStencilAttachment = {
    label: "depth stencil attachment",
    view: targetViews.depth,
    depthClearValue: 1,
    depthLoadOp: "clear",
    depthStoreOp: "store",
} as GPURenderPassDepthStencilAttachment;

const normalTargetAttachment: GPURenderPassColorAttachment = {
    label: "normal target attachment",
    view: targetViews.normal,
    clearValue: [0, 0, 0, 1],
    loadOp: "clear",
    storeOp: "store",
} as GPURenderPassColorAttachment;

const positionTargetAttachment: GPURenderPassColorAttachment = {
    label: "position target attachment",
    view: targetViews.position,
    clearValue: [0, 0, 0, 1],
    loadOp: "clear",
    storeOp: "store",
} as GPURenderPassColorAttachment;

const targetPassDescriptor: GPURenderPassDescriptor = {
    label: "target render pass",
    colorAttachments: [
        colorTargetAttachment,
        normalTargetAttachment,
        positionTargetAttachment,
    ],
    depthStencilAttachment: depthStencilAttachment,
    timestampWrites: targetGPUTiming.timestampWrites,
} as GPURenderPassDescriptor;

const finalColorAttachment: GPURenderPassColorAttachment = {
    label: "final color attachment",
    view: context!.getCurrentTexture().createView(),
    clearValue: [0.3, 0.3, 0.3, 1],
    loadOp: "clear",
    storeOp: "store",
} as GPURenderPassColorAttachment;

const deferredPassDescriptor: GPURenderPassDescriptor = {
    label: "deferred render pass",
    colorAttachments: [finalColorAttachment],
    timestampWrites: deferredGPUTiming.timestampWrites,
} as GPURenderPassDescriptor;

//////////// CREATE BUNDLE ////////////

const shadowBundleEncoder: GPURenderBundleEncoder =
    device.createRenderBundleEncoder({
        label: "shadow bundle",
        colorFormats: [],
        depthStencilFormat: shadowFormat,
    } as GPURenderBundleEncoderDescriptor);
shadowBundleEncoder.setPipeline(shadowPipeline);
shadowBundleEncoder.setBindGroup(0, shadowBindGroup);
shadowBundleEncoder.setIndexBuffer(indexBuffer, "uint32");
geometries.forEach((_geometry: OBJParseResult, i: int) => {
    shadowBundleEncoder.drawIndexedIndirect(indirectBuffer, 20 * i);
});
const shadowBundle: GPURenderBundle = shadowBundleEncoder.finish();

const targetBundleEncoder: GPURenderBundleEncoder =
    device.createRenderBundleEncoder({
        label: "target bundle",
        colorFormats: [colorFormat, normalFormat, positionFormat],
        depthStencilFormat: depthFormat,
    } as GPURenderBundleEncoderDescriptor);
targetBundleEncoder.setPipeline(targetPipeline);
targetBundleEncoder.setBindGroup(0, targetBindGroup);
targetBundleEncoder.setIndexBuffer(indexBuffer, "uint32");
geometries.forEach((_geometry: OBJParseResult, i: int) => {
    targetBundleEncoder.drawIndexedIndirect(indirectBuffer, 20 * i);
});
const targetBundle: GPURenderBundle = targetBundleEncoder.finish();

const deferredBundleEncoder: GPURenderBundleEncoder =
    device.createRenderBundleEncoder({
        label: "deferred bundle",
        colorFormats: [presentationFormat],
    } as GPURenderBundleEncoderDescriptor);
deferredBundleEncoder.setPipeline(deferredPipeline);
deferredBundleEncoder.setBindGroup(0, deferredBindGroup);
deferredBundleEncoder.draw(6);
const deferredBundle: GPURenderBundle = deferredBundleEncoder.finish();

//////////// EACH FRAME ////////////

async function frame(now: float): Promise<void> {
    stats.time("cpu delta");

    //////////// UPDATE CONTROL CAMERA UNIFORMS ////////////

    control.update();
    camera.update().store(uniformData, 0);
    device!.queue.writeBuffer(uniformBuffer, 0, uniformData);

    //////////// RENDER FRAME ////////////

    finalColorAttachment.view = context!.getCurrentTexture().createView();

    const renderEncoder: GPUCommandEncoder = device!.createCommandEncoder({
        label: "render command encoder",
    } as GPUObjectDescriptorBase);

    const shadowPass: GPURenderPassEncoder =
        renderEncoder.beginRenderPass(shadowPassDescriptor);
    shadowPass.executeBundles([shadowBundle]);
    shadowPass.end();

    const targetPass: GPURenderPassEncoder =
        renderEncoder.beginRenderPass(targetPassDescriptor);
    targetPass.executeBundles([targetBundle]);
    targetPass.end();

    const deferredPass: GPURenderPassEncoder = renderEncoder.beginRenderPass(
        deferredPassDescriptor,
    );
    deferredPass.executeBundles([deferredBundle]);
    deferredPass.end();

    shadowGPUTiming.resolve(renderEncoder);
    targetGPUTiming.resolve(renderEncoder);
    deferredGPUTiming.resolve(renderEncoder);

    const renderCommandBuffer: GPUCommandBuffer = renderEncoder.finish();
    device!.queue.submit([renderCommandBuffer]);

    //////////// UPDATE STATS ////////////

    stats.time("cpu delta", "cpu delta");
    cpuDelta.sample(stats.get("cpu delta")!);

    stats.set("frame delta", now - stats.get("frame delta")!);
    frameDelta.sample(stats.get("frame delta")!);

    shadowGPUTiming.readback((ms: float) => {
        stats.set("gpu shadow delta", ms);
        gpuShadowDelta.sample(ms);
    });
    targetGPUTiming.readback((ms: float) => {
        stats.set("gpu target delta", ms);
        gpuTargetDelta.sample(ms);
    });
    deferredGPUTiming.readback((ms: float) => {
        stats.set("gpu deferred delta", ms);
        gpuDeferredDelta.sample(ms);
    });
    const gpuDeltaSum: float =
        gpuShadowDelta.get() + gpuTargetDelta.get() + gpuDeferredDelta.get();

    // prettier-ignore
    stats.update(`
        <b>frame rate: ${(1_000 / frameDelta.get()).toFixed(0)} fps</b><br>
        frame delta: ${frameDelta.get().toFixed(2)} ms<br>
        <br>
        <b>cpu rate: ${(1_000 / cpuDelta.get()).toFixed(0)} fps</b><br>
        cpu delta: ${cpuDelta.get().toFixed(2)} ms<br>
        <br>
        <b>gpu rate: ${(1_000 / gpuDeltaSum).toFixed(0)} fps</b><br>
        gpu delta: ${gpuDeltaSum.toFixed(2)} ms<br>
         - shadow delta: ${gpuShadowDelta.get().toFixed(2)} ms<br>
         - target delta: ${gpuTargetDelta.get().toFixed(2)} ms<br>
         - deferred delta: ${gpuDeferredDelta.get().toFixed(2)} ms<br>
    `);
    stats.set("frame delta", now);

    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
