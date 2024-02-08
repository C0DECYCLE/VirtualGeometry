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

struct ClusterChildren {
    length: u32,
    ids: array<ClusterId, 2>
}

struct Cluster {
    error: f32,
    parentError: f32,
    children: ClusterChildren
};

struct Indirect {
    vertexCount: u32,
    instanceCount: atomic<u32>,
    firstVertex: u32,
    firstInstance: u32
};

struct Queue {
    length: atomic<u32>,
    queue: array<ClusterId>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> queue: Queue;
@group(0) @binding(2) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(3) var<storage, read_write> indirect: Indirect;
@group(0) @binding(4) var<storage, read_write> clusterDraw: array<ClusterId>;

@compute @workgroup_size(1, 1, 1) fn cs(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
    let threshold: f32 = (length(uniforms.cameraPosition) - 1) * 0.1; // (length(camera-objectposition) - objectradius) * 0... //compute in instance compute shader and pass here
    
    while(true) {

        let length: u32 = atomicLoad(&queue.length);
        if (length == 0) {
            break;
        }
        if (!atomicCompareExchangeWeak(&queue.length, length, length - 1).exchanged) {
            continue;
        }

        let id: ClusterId = queue.queue[length - 1];
        let cluster: Cluster = clusters[id];

        if ((cluster.parentError == 0 || cluster.parentError > threshold) && 
            (cluster.children.length == 0 || cluster.error <= threshold)) {
            clusterDraw[atomicAdd(&indirect.instanceCount, 1)] = id;
            continue;
        }

        for (var i: u32 = 0; i < cluster.children.length; i++) {
            queue.queue[atomicAdd(&queue.length, 1)] = cluster.children.ids[i];
        }
    }
}
