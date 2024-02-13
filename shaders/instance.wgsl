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

struct AtomicQueue {
    front: atomic<u32>,
    rear: atomic<u32>,
    items: array<DrawPair>
};

struct Indirect {
    workgroupCount: atomic<u32>
}

@group(0) @binding(0) var<storage, read> entities: array<Entity>;
@group(0) @binding(1) var<storage, read_write> atomicQueue: AtomicQueue;
@group(0) @binding(2) var<storage, read_write> indirect: Indirect;

@compute @workgroup_size(1) fn cs(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
    let index: u32 = globalInvocationId.x;
    let entity: Entity = entities[index];
    enqueue(DrawPair(index, entity.root));
    increaseIndirect();
}

fn enqueue(drawPair: DrawPair) {
    atomicQueue.items[atomicAdd(&atomicQueue.rear, 1)] = drawPair;
}

fn increaseIndirect() {
    atomicAdd(&indirect.workgroupCount, 1);
}