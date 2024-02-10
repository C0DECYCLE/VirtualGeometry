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
    root: ClusterId
};

struct Queue {
    lock: atomic<u32>, // 0 = no-lock
    length: atomic<u32>,
    queue: array<DrawPair>,
};


@group(0) @binding(0) var<storage, read> entities: array<Entity>;
@group(0) @binding(1) var<storage, read_write> queue: Queue;

@compute @workgroup_size(1) fn cs(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
    let index: u32 = globalInvocationId.x;
    let entity: Entity = entities[index];
    queuePush(DrawPair(index, entity.root));
}

fn queuePush(drawPair: DrawPair) {
    queue.queue[atomicAdd(&queue.length, 1)] = drawPair;
}
