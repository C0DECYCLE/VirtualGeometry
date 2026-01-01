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

fn pack_drawPair(drawPair: DrawPair) -> u32 {
    return (drawPair.entity << 16) | (drawPair.cluster & 0xFFFF);
}

fn unpack_drawPair(packed: u32) -> DrawPair {
    let entity = packed >> 16;
    let cluster = packed & 0xFFFF;
    return DrawPair(entity, cluster);
}

struct Uniforms {
    viewProjection: mat4x4f,
    viewMode: u32,
    cameraPosition: vec3f
};

struct Children {
    length: u32,
    ids: array<ClusterId, 2>
}

struct Cluster {
    error: f32,
    parentError: f32,
    children: Children
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

const QUEUE_CAPACITY: u32 = 1024 * 1024; // * 32
const QUEUE_UNUSED: u32 = 0xFFFFFFFF;

struct Queue {
    pad1: array<u32, 31>,
    head: atomic<u32>,

    pad2: array<u32, 31>,
    tail: atomic<u32>,

    pad3: array<u32, 31>,
    count: atomic<i32>,

    ring: array<atomic<u32>>,
}

fn ensure_enqueue() -> bool {
    if (atomicLoad(&queue.count) >= i32(QUEUE_CAPACITY)) { 
        return false; 
    }
    let prev: i32 = atomicAdd(&queue.count, 1);
    if (prev < i32(QUEUE_CAPACITY)) { 
        return true; 
    }
    atomicSub(&queue.count, 1);
    return false;
}

fn ensure_dequeue() -> bool {
    if (atomicLoad(&queue.count) <= 0) { 
        return false; 
    }
    let prev: i32 = atomicSub(&queue.count, 1);
    if (prev > 0) { 
        return true; 
    }
    atomicAdd(&queue.count, 1);
    return false;
}

fn publish_slot(p: u32, data: u32) {
    loop {
        let r = atomicCompareExchangeWeak(&queue.ring[p], QUEUE_UNUSED, data);
        if (r.exchanged) { 
            break; 
        }
    }
}

fn consume_slot(p: u32) -> u32 {
    loop {
        let val: u32 = atomicExchange(&queue.ring[p], QUEUE_UNUSED);
        if (val != QUEUE_UNUSED) {
            return val;
        }
    }
}

fn enqueue(data: u32) -> bool {
    if (data == QUEUE_UNUSED) { 
        return false; 
    }
    if (!ensure_enqueue()) { 
        return false; 
    }
    let pos: u32 = atomicAdd(&queue.tail, 1);
    let p: u32 = pos % QUEUE_CAPACITY;
    publish_slot(p, data);
    return true;
}

fn dequeue(data: ptr<function, u32>) -> bool {
    if (!ensure_dequeue()) { 
        return false; 
    }
    let pos: u32 = atomicAdd(&queue.head, 1);
    let p: u32 = pos % QUEUE_CAPACITY;
    (*data) = consume_slot(p);
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
        if (!dequeue(&data)) { 
            break; 
        }
        let drawPair: DrawPair = unpack_drawPair(data);
        let cluster: Cluster = clusters[drawPair.cluster];
        let entity: Entity = entities[drawPair.entity];
        let distance: f32 = length(entity.position - uniforms.cameraPosition);
        let threshold: f32 = (distance - entity.radius) * THRESHOLD_SCALE;
        let parentOverThreshold: bool = cluster.parentError == 0 || cluster.parentError > threshold;
        let clusterUnderThreshold: bool = cluster.children.length == 0 || cluster.error <= threshold;
        if (parentOverThreshold && clusterUnderThreshold) {
            drawPairs[atomicAdd(&indirect.instanceCount, 1)] = drawPair;
            continue;
        }
        for (var i: u32 = 0; i < cluster.children.length; i++) {
            let nDrawPair: DrawPair = DrawPair(drawPair.entity, cluster.children.ids[i]);
            let packed: u32 = pack_drawPair(nDrawPair);
            loop { 
                if (enqueue(packed)) { 
                    break; 
                } 
            }
        }
    }
}