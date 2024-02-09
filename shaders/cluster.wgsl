/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

alias EntityIndex = u32;

alias ClusterId = u32;

struct DrawPair {
    entity: EntityIndex,
    cluster: ClusterId
};

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

struct Entity {
    position: vec3f,
    root: ClusterId
};

struct Indirect {
    vertexCount: u32,
    instanceCount: atomic<u32>,
    firstVertex: u32,
    firstInstance: u32
};

struct Queue {
    length: atomic<u32>,
    queue: array<DrawPair>,
};

override WORKGROUP_SIZE: u32;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> queue: Queue;
@group(0) @binding(2) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(3) var<storage, read> entities: array<Entity>;
@group(0) @binding(4) var<storage, read_write> indirect: Indirect;
@group(0) @binding(5) var<storage, read_write> drawPairs: array<DrawPair>;

@compute @workgroup_size(WORKGROUP_SIZE) fn cs() {
    while(true) {

        let len: u32 = atomicLoad(&queue.length);
        if (len == 0) {
            break;
        }

        if (!atomicCompareExchangeWeak(&queue.length, len, len - 1).exchanged) {
            continue;
        }
        
        let drawPair: DrawPair = queue.queue[len - 1];
        let cluster: Cluster = clusters[drawPair.cluster];
        let entity: Entity = entities[drawPair.entity];
        let threshold: f32 = (length(entity.position - uniforms.cameraPosition) - 1) * 0.1; 
        // - entity/geometry.radius

        if ((cluster.parentError == 0 || cluster.parentError > threshold) && 
            (cluster.children.length == 0 || cluster.error <= threshold)) {
            drawPairs[atomicAdd(&indirect.instanceCount, 1)] = drawPair;
            continue;
        }

        for (var i: u32 = 0; i < cluster.children.length; i++) {
            queue.queue[atomicAdd(&queue.length, 1)] = DrawPair(drawPair.entity, cluster.children.ids[i]);
        }
    }
}
