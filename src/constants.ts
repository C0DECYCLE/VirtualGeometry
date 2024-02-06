/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { int } from "./utils.type.js";

export const Bytes4: int = 4;

export const Vec3Layout: int = 3 + 1;

export const WebGPURequirements: GPUFeatureName[] = [
    "timestamp-query",
    "indirect-first-instance",
];

export const ClusterTrianglesLimit: int = 128;

export const ClusterGroupingLimit: int = 4;

export const ClusterLayout: int = 3 + 1 + 3 + 2 + 4 + 1 + 2;

export const TasksLimit: int = 100;

export const VertexStride: int = Vec3Layout;

export const AnalyticSamples: int = 60;

export const UniformsLayout: int = 4 * 4 + 1 + 1 + 2 + 3 + 1;

export const EntityStride: int = Vec3Layout;

export const EntityLimit: int = 100;

export const TextureFormats = {
    depth: "depth24plus",
};

export const ShaderPaths = {
    evaluation: "./shaders/evaluation.wgsl",
    draw: "./shaders/draw.wgsl",
};
