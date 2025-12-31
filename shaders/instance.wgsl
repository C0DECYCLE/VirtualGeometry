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

struct Entity {
    position: vec3f,
    root: ClusterId,
    radius: f32
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

fn ensure_enqueue(q: ptr<storage, Queue, read_write>) -> bool {
    if (atomicLoad(&(*q).count) >= i32(QUEUE_CAPACITY)) { 
        return false; 
    }
    let prev: i32 = atomicAdd(&(*q).count, 1);
    if (prev < i32(QUEUE_CAPACITY)) { 
        return true; 
    }
    atomicSub(&(*q).count, 1);
    return false;
}

fn ensure_dequeue(q: ptr<storage, Queue, read_write>) -> bool {
    if (atomicLoad(&(*q).count) <= 0) { 
        return false; 
    }
    let prev: i32 = atomicSub(&(*q).count, 1);
    if (prev > 0) { 
        return true; 
    }
    atomicAdd(&(*q).count, 1);
    return false;
}

fn publish_slot(q: ptr<storage, Queue, read_write>, p: u32, data: u32) {
    loop {
        let r = atomicCompareExchangeWeak(&(*q).ring[p], QUEUE_UNUSED, data);
        if (r.exchanged) { 
            break; 
        }
    }
}

fn consume_slot(q: ptr<storage, Queue, read_write>, p: u32) -> u32 {
    loop {
        let val: u32 = atomicExchange(&(*q).ring[p], QUEUE_UNUSED);
        if (val != QUEUE_UNUSED) {
            return val;
        }
    }
}

fn enqueue(q: ptr<storage, Queue, read_write>, data: u32) -> bool {
    if (data == QUEUE_UNUSED) { 
        return false; 
    }
    if (!ensure_enqueue(q)) { 
        return false; 
    }
    let pos: u32 = atomicAdd(&(*q).tail, 1);
    let p: u32 = pos % QUEUE_CAPACITY;
    publish_slot(q, p, data);
    return true;
}

fn dequeue(q: ptr<storage, Queue, read_write>, out_data: ptr<function, u32>) -> bool {
    if (!ensure_dequeue(q)) { 
        return false; 
    }
    let pos: u32 = atomicAdd(&(*q).head, 1);
    let p: u32 = pos % QUEUE_CAPACITY;
    (*out_data) = consume_slot(q, p);
    return true;
}

const WORKGROUP_SIZE: u32 = 256;

@group(0) @binding(0) var<storage, read> entities: array<Entity>;
@group(0) @binding(1) var<storage, read_write> queue: Queue;

@compute @workgroup_size(WORKGROUP_SIZE) fn cs(
    @builtin(global_invocation_id) globalInvocationId: vec3<u32>
) {
    let index: u32 = globalInvocationId.x;
    if (index >= arrayLength(&entities)) {
        return;
    }
    let entity: Entity = entities[index];
    let drawPair: DrawPair = DrawPair(index, entity.root);
    enqueue(&queue, pack_drawPair(drawPair));
}