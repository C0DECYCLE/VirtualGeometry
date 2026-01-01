/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { int } from "./utils.type.js";

export const Bytes4: int = 4;

export const VertexLayout: int = 3 + 1;
export const EntityLayout: int = 3 + 1 + 1 + 3;
export const UniformsLayout: int = 4 * 4 + 1 + 3 + 3 + 1;
export const ClusterLayout: int = 1 + 1 + 1 + 1 + 1;
export const DrawPairLayout: int = 1 + 1;
export const QueueHeaderLayout: int = 32 * 3;

export const AllowMicroCracks: boolean = true;

export const ClusterTrianglesLimit: int = 128;
export const ClusterGroupingLimit: int = 4;
export const EntityLimit: int = 10_000;

export const PersistentThreadGroups: int = 64;
export const PersistentThreadsPerGroup: int = 256;

export const QueueLimit: int = 1024 * 128; // 1024 * 1024 * 32
export const DrawPairLimit: int = 100_000;

export const AnalyticSamples: int = 60;

export const WebGPURequirements: GPUFeatureName[] = [
    "timestamp-query",
    "indirect-first-instance",
    "primitive-index",
];

export const WebGPULimits: Record<string, GPUSize64> = {
    //maxStorageBufferBindingSize: (QueueHeaderLayout + QueueLimit) * Bytes4,
};

export const TextureFormats = {
    depth: "depth32float",
};

export const ShaderPaths = {
    instance: "./shaders/instance.wgsl",
    cluster: "./shaders/cluster.wgsl",
    draw: "./shaders/draw.wgsl",
};
