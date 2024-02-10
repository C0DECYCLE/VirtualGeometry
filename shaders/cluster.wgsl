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
    root: ClusterId
};

struct Indirect {
    vertexCount: u32,
    instanceCount: atomic<u32>,
    firstVertex: u32,
    firstInstance: u32
};

struct Queue {
    lock: atomic<u32>, // 0 = no-lock
    length: atomic<u32>,
    queue: array<DrawPair>
};

override WORKGROUP_SIZE: u32;
const THRESHOLD_SCALE: f32 = 0.1;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> queue: Queue;
@group(0) @binding(2) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(3) var<storage, read> entities: array<Entity>;
@group(0) @binding(4) var<storage, read_write> indirect: Indirect;
@group(0) @binding(5) var<storage, read_write> drawPairs: array<DrawPair>;

@compute @workgroup_size(WORKGROUP_SIZE) fn cs(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
    //let id: u32 = globalInvocationId.x + 1; // to ensure no 0 for lock

    var safety: u32 = 0;
    while(safety < 1000000) {
        safety += 1;

        /*
        if (queueRequestLock(id)) {

            if (atomicLoad(&queue.length) == 0) {
                break;
            }
            let drawPair: DrawPair = queuePop();

            let cluster: Cluster = clusters[drawPair.cluster];
            let entity: Entity = entities[drawPair.entity];
            let threshold: f32 = (length(entity.position - uniforms.cameraPosition) - 1) * THRESHOLD_SCALE; // - entity/geometry.radius

            if ((cluster.parentError == 0 || cluster.parentError > threshold) && (cluster.children.length == 0 || cluster.error <= threshold)) {
                drawPairsPush(drawPair);
            } else {
                for (var i: u32 = 0; i < cluster.children.length; i++) {
                    queuePush(DrawPair(drawPair.entity, cluster.children.ids[i]));
                }
            }
            
            queueUnLock();
        }
        */

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
        let threshold: f32 = (length(entity.position - uniforms.cameraPosition) - 1) * THRESHOLD_SCALE; // - entity/geometry.radius

        if ((cluster.parentError == 0 || cluster.parentError > threshold) && (cluster.children.length == 0 || cluster.error <= threshold)) {
            drawPairsPush(drawPair);
        } else {
            for (var i: u32 = 0; i < cluster.children.length; i++) {
                queuePush(DrawPair(drawPair.entity, cluster.children.ids[i]));
            }
        }
    }
}

fn queueRequestLock(id: u32) -> bool {
    return atomicCompareExchangeWeak(&queue.lock, 0, id).exchanged;
}

fn queueUnLock() {
    atomicStore(&queue.lock, 0);
}

fn queuePush(drawPair: DrawPair) {
    queue.queue[atomicAdd(&queue.length, 1)] = drawPair;
}

fn queuePop() -> DrawPair {
    return queue.queue[atomicSub(&queue.length, 1) - 1];
}

fn drawPairsPush(drawPair: DrawPair) {
    drawPairs[atomicAdd(&indirect.instanceCount, 1)] = drawPair;
}