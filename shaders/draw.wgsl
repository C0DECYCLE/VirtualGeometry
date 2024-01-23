/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

struct Uniforms {
    viewProjection: mat4x4f,
};

struct Vertex {
    position: vec3f,
};

struct VertexShaderOut {
    @builtin(position) position: vec4f,
    @interpolate(flat) @location(0) color: vec3f
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> vertecies: array<Vertex>;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) clusterIndex: u32) -> VertexShaderOut {
    let index: u32 = indices[clusterIndex * 128 * 3 + vertexIndex];
    let vertex: Vertex = vertecies[index];

    var position: vec3f = vertex.position;

    var out: VertexShaderOut;
    out.position = uniforms.viewProjection * vec4f(position, 1);
    let uid: f32 = f32(clusterIndex) + 1;
    out.color = fract(vec3f(uid * 0.1443, uid * 0.6841, uid * 0.7323));
    return out;
}

@fragment fn fs(in: VertexShaderOut) -> @location(0) vec4f {
    return vec4f(in.color, 1);
}