/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { int } from "./utils.type.js";

export const Bytes4: int = 4;

export const Vec3Layout: int = 3 + 1;

export const VertexStride: int = Vec3Layout;

export const EntityStride: int = Vec3Layout;

export const UniformsLayout: int = 4 * 4 + 1 + 3 + 3 + 1;

export const ClusterLayout: int = 1 + 1 + 1 + 1 + 1;

export const AllowMicroCracks: boolean = true;

export const ClusterTrianglesLimit: int = 128;

export const ClusterGroupingLimit: int = 4; //dont change because of tree extraction

export const EntityLimit: int = 100;

export const PersistentThreads: int = 256;

export const QueueLimit: int = EntityLimit * 10;

export const ClusterDrawLimit: int = 1000;

export const AnalyticSamples: int = 60;

export const WebGPURequirements: GPUFeatureName[] = [
    "timestamp-query",
    "indirect-first-instance",
];

export const TextureFormats = {
    depth: "depth24plus",
};

export const ShaderPaths = {
    instance: "./shaders/instance.wgsl",
    cluster: "./shaders/cluster.wgsl",
    draw: "./shaders/draw.wgsl",
};
