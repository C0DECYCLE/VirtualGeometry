/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 * Queue by Kavosh Jazar, December 2025
 */

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
    let packed: u32 = pack_drawPair(drawPair);
    enqueue(packed);
}