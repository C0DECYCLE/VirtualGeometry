/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

struct Uniforms {
    viewProjection: mat4x4f,
    viewMode: u32,
    cameraPosition: vec3f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(1, 1, 1) fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
    let threshold: f32 = length(uniforms.cameraPosition) * 0.05; // (pow(length(camera-objectposition)) - objectradius) * 0.05 //compute in instance compute shader and pass here
}
