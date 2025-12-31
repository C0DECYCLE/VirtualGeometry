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
    cameraPosition: vec3f
};

struct Tree {
    length: u32,
    ids: array<ClusterId, 2>
}

struct Cluster {
    error: f32,
    parentError: f32,
    children: Tree
};

struct Entity {
    position: vec3f,
    root: ClusterId,
    radius: f32
};

struct Indirect {
    vertexCount: u32,
    instanceCount: atomic<u32>,
    firstVertex: u32,
    firstInstance: u32
};

struct Queue {
    front: atomic<u32>,
    rear: atomic<u32>,
    items: array<u32>
};

fn enqueue(value: u32) {
    queue.items[atomicAdd(&queue.rear, 1)] = value;
}

fn dequeue() -> u32 {
    return queue.items[atomicAdd(&queue.front, 1)];
}

override WORKGROUP_SIZE: u32;
const THRESHOLD_SCALE: f32 = 0.1;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> queue: Queue;
@group(0) @binding(2) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(3) var<storage, read> entities: array<Entity>;
@group(0) @binding(4) var<storage, read_write> indirect: Indirect;
@group(0) @binding(5) var<storage, read_write> drawPairs: array<DrawPair>;

@compute @workgroup_size(WORKGROUP_SIZE) fn cs() {
    while(atomicLoad(&queue.front) < atomicLoad(&queue.rear)) {
        let drawPair: DrawPair = unpack_drawPair(dequeue());
        let cluster: Cluster = clusters[drawPair.cluster];
        let entity: Entity = entities[drawPair.entity];
        let threshold: f32 = (length(entity.position - uniforms.cameraPosition) - entity.radius) * THRESHOLD_SCALE;
        if ((cluster.parentError == 0 || cluster.parentError > threshold) && (cluster.children.length == 0 || cluster.error <= threshold)) {
            drawPairsPush(drawPair);
        } else {
            for (var i: u32 = 0; i < cluster.children.length; i++) {
                let nDrawPair: DrawPair = DrawPair(drawPair.entity, cluster.children.ids[i]);
                enqueue(pack_drawPair(nDrawPair));
            }
        }
    }
}

fn pack_drawPair(drawPair: DrawPair) -> u32 {
    return (drawPair.entity << 16u) | (drawPair.cluster & 0xFFFFu);
}

fn unpack_drawPair(packed: u32) -> DrawPair {
    let entity = packed >> 16u;
    let cluster = packed & 0xFFFFu;
    return DrawPair(entity, cluster);
}

fn drawPairsPush(drawPair: DrawPair) {
    drawPairs[atomicAdd(&indirect.instanceCount, 1)] = drawPair;
}