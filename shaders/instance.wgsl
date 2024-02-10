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
    front: atomic<u32>,
    rear: atomic<u32>,
    items: array<DrawPair>
};

@group(0) @binding(0) var<storage, read> entities: array<Entity>;
@group(0) @binding(1) var<storage, read_write> queue: Queue;

@compute @workgroup_size(1) fn cs(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
    let index: u32 = globalInvocationId.x;
    let entity: Entity = entities[index];
    enqueue(DrawPair(index, entity.root));
}

fn enqueue(drawPair: DrawPair) {
    queue.items[atomicAdd(&queue.rear, 1)] = drawPair;
}