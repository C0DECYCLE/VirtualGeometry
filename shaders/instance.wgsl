/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

alias ClusterId = u32;

struct Uniforms {
    viewProjection: mat4x4f,
    viewMode: u32,
    cameraPosition: vec3f,
};

struct Queue {
    length: atomic<u32>,
    queue: array<ClusterId>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> queue: Queue;

@compute @workgroup_size(1, 1, 1) fn cs(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
    for (var i: u32 = 0; i < 275; i++) {
        queue.queue[atomicAdd(&queue.length, 1)] = i;
    }
    let threshold: f32 = length(uniforms.cameraPosition) * 0.05; // (pow(length(camera-objectposition)) - objectradius) * 0.05 //compute in instance compute shader and pass here
}
