/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 * Queue by Kavosh Jazar, December 2025
 */

requires unrestricted_pointer_parameters;

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

const QUEUE_CAPACITY: u32 = 1024 * 1024 * 32;
const QUEUE_UNUSED: u32 = 0xFFFFFFFFu;

struct Queue {
    pad1: array<u32, 31>,
    head: atomic<u32>,

    pad2: array<u32, 31>,
    tail: atomic<u32>,

    pad3: array<u32, 31>,
    count: atomic<i32>,

    ring: array<atomic<u32>>,
}

fn size(q: ptr<storage, Queue, read_write>) -> i32 {
    return atomicLoad(&(*q).count);
}

fn ensure_enqueue(q: ptr<storage, Queue, read_write>) -> bool {
    if (atomicLoad(&(*q).count) >= i32(QUEUE_CAPACITY)) { return false; }

    let prev = atomicAdd(&(*q).count, 1);
    if (prev < i32(QUEUE_CAPACITY)) { return true; }
    atomicSub(&(*q).count, 1);
    return false;
}

fn ensure_dequeue(q: ptr<storage, Queue, read_write>) -> bool {
    if (atomicLoad(&(*q).count) <= 0) { return false; }

    let prev = atomicSub(&(*q).count, 1);
    if (prev > 0) { return true; }
    atomicAdd(&(*q).count, 1);
    return false;
}

fn publish_slot(q: ptr<storage, Queue, read_write>, p: u32, data: u32) {
    loop {
        let r = atomicCompareExchangeWeak(&(*q).ring[p], QUEUE_UNUSED, data);
        if (r.exchanged) { break; }
    }
}

fn consume_slot(q: ptr<storage, Queue, read_write>, p: u32) -> u32 {
    loop {
        let val = atomicExchange(&(*q).ring[p], QUEUE_UNUSED);
        if (val != QUEUE_UNUSED) {
            return val;
        }
    }
}

fn enqueue(q: ptr<storage, Queue, read_write>, data: u32) -> bool {
    if (data == QUEUE_UNUSED) { return false; }

    if (!ensure_enqueue(q)) { return false; }

    let pos = atomicAdd(&(*q).tail, 1u);
    let p = pos % QUEUE_CAPACITY;

    publish_slot(q, p, data);

    return true;
}

fn dequeue(q: ptr<storage, Queue, read_write>, out_data: ptr<function, u32>) -> bool {
    if (!ensure_dequeue(q)) { return false; }

    let pos = atomicAdd(&(*q).head, 1u);
    let p = pos % QUEUE_CAPACITY;

    (*out_data) = consume_slot(q, p);

    return true;
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
    loop {
        var data: u32;
        if (!dequeue(&queue, &data)) { break; }
        let drawPair: DrawPair = unpack_drawPair(data);
        let cluster: Cluster = clusters[drawPair.cluster];
        let entity: Entity = entities[drawPair.entity];
        let threshold: f32 = (length(entity.position - uniforms.cameraPosition) - entity.radius) * THRESHOLD_SCALE;
        if ((cluster.parentError == 0 || cluster.parentError > threshold) && (cluster.children.length == 0 || cluster.error <= threshold)) {
            drawPairsPush(drawPair);
        } else {
            for (var i: u32 = 0; i < cluster.children.length; i++) {
                let nDrawPair: DrawPair = DrawPair(drawPair.entity, cluster.children.ids[i]);
                loop { if (enqueue(&queue, pack_drawPair(nDrawPair))) { break; } }
                // if full retry until free slot, ideally this is never the case if the queue is big enough
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