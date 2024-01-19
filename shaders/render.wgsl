/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

struct Uniforms {
    viewProjection: mat4x4f,
};

struct Index {
    index: u32,
    triangleId: u32,
    clusterId: u32,
}

struct Vertex {
    position: vec3f,
};

struct Instance {
    position: vec3f,
};

struct VertexShaderOut {
    @builtin(position) position: vec4f,
    @interpolate(flat) @location(0) color: vec3f
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> indices: array<Index>;
@group(0) @binding(2) var<storage, read> vertecies: array<Vertex>;
@group(0) @binding(3) var<storage, read> instances: array<Instance>;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexShaderOut {
    let index: Index = indices[vertexIndex];
    let vertex: Vertex = vertecies[index.index];
    let instance: Instance = instances[instanceIndex];

    let position: vec3f = vertex.position + instance.position;
    
    var out: VertexShaderOut;
    out.position = uniforms.viewProjection * vec4f(position, 1);
    let uid: f32 = f32(index.clusterId) + 1; //f32(vertexIndex);
    out.color = fract(vec3f(uid * 0.1443, uid * 0.6841, uid * 0.7323));
    return out;
}

@fragment fn fs(in: VertexShaderOut) -> @location(0) vec4f {
    return vec4f(in.color, 1);
}