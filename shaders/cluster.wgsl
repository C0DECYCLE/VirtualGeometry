/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

alias ClusterId = u32;

struct Uniforms {
    viewProjection: mat4x4f,
    viewMode: u32,
    cameraPosition: vec3f,
};

struct Cluster {
    error: f32,
    parentError: f32,
    parentsLength: u32,
    childrenLength: u32
};

struct Indirect {
    vertexCount: u32,
    instanceCount: atomic<u32>,
    firstVertex: u32,
    firstInstance: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read_write> clusterDraw: array<ClusterId>;
@group(0) @binding(3) var<storage, read_write> indirect: Indirect;

@compute @workgroup_size(1, 1, 1) fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
    let threshold: f32 = length(uniforms.cameraPosition) * 0.05; // (pow(length(camera-objectposition)) - objectradius) * 0.05 //compute in instance compute shader and pass here
    let cluster: Cluster = clusters[id.x];
    if ((cluster.parentsLength == 0 || cluster.parentError > threshold) && (cluster.childrenLength == 0 || cluster.error <= threshold)) {
        clusterDraw[atomicAdd(&indirect.instanceCount, 1)] = id.x;
    }
}
