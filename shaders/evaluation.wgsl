/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

alias ClusterId = u32;

struct Uniforms {
    viewProjection: mat4x4f,
    viewMode: u32,
    resolution: vec2f,
    cameraPosition: vec3f,
};

struct Cluster {
    min: vec3f,
    max: vec3f,
    parents: array<ClusterId, 2>,
    children: array<ClusterId, 4>,
    error: f32
};

struct Indirect {
    invokeCount: u32,
    taskCount: atomic<u32>,
    firstInvoke: u32,
    firstTask: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read_write> tasks: array<ClusterId>;
@group(0) @binding(3) var<storage, read_write> indirect: Indirect;

@compute @workgroup_size(256, 1, 1) fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
    let threshold: f32 = length(uniforms.cameraPosition) * 0.1;

    let cluster: Cluster = clusters[id.x];
    let clusterError: f32 = cluster.error;

    var childrenLength: u32 = 0;
    if (cluster.children[0] != 0) {
        childrenLength += 1;
    } 
    if (cluster.children[1] != 0) {
        childrenLength += 1;
    } 
    if (cluster.children[2] != 0) {
        childrenLength += 1;
    } 
    if (cluster.children[3] != 0) {
        childrenLength += 1;
    } 

    var parentsLength: u32 = 0;
    var parentError: f32 = 0;
    if (cluster.parents[0] != 0) {
        parentError = clusters[cluster.parents[0] - 1].error;
        parentsLength += 1;
    } 
    if (cluster.parents[1] != 0) {
        parentError = clusters[cluster.parents[1] - 1].error;
        parentsLength += 1;
    } 

    if ((parentError > threshold || parentsLength == 0) && (clusterError <= threshold || childrenLength == 0)) {
        tasks[atomicAdd(&indirect.taskCount, 1)] = id.x;
    }
}
