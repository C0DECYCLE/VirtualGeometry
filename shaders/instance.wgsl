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

struct Entity {
    position: vec3f,
    root: ClusterId,
    radius: f32
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

@group(0) @binding(0) var<storage, read> entities: array<Entity>;
@group(0) @binding(1) var<storage, read_write> queue: Queue;

@compute @workgroup_size(256) fn cs(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
    let index: u32 = globalInvocationId.x;
    if (index >= arrayLength(&entities)) {
        return;
    }
    let entity: Entity = entities[index];
    let drawPair: DrawPair = DrawPair(index, entity.root);
    enqueue(pack_drawPair(drawPair));
}

fn pack_drawPair(drawPair: DrawPair) -> u32 {
    return (drawPair.entity << 16u) | (drawPair.cluster & 0xFFFFu);
}

fn unpack_drawPair(packed: u32) -> DrawPair {
    let entity = packed >> 16u;
    let cluster = packed & 0xFFFFu;
    return DrawPair(entity, cluster);
}