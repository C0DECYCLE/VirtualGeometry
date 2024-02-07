/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

alias ClusterId = u32;
alias VertexId = u32;

struct Uniforms {
    viewProjection: mat4x4f,
    viewMode: u32,
    cameraPosition: vec3f,
};

struct Vertex {
    position: vec3f,
};

struct VertexShaderOut {
    @builtin(position) clipspace: vec4f,
    @interpolate(flat) @location(0) color: vec3f,
    @location(1) position: vec3f
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> clusterDraw: array<ClusterId>;
@group(0) @binding(2) var<storage, read> triangles: array<VertexId>;
@group(0) @binding(3) var<storage, read> vertecies: array<Vertex>;

@vertex fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexShaderOut {
    let clusterId: ClusterId = clusterDraw[instanceIndex];
    let vertexId: VertexId = triangles[clusterId * 128 * 3 + vertexIndex];
    let vertex: Vertex = vertecies[vertexId];
    var position: vec3f = vertex.position;
    var out: VertexShaderOut;
    out.clipspace = uniforms.viewProjection * vec4f(position, 1);
    out.color = getColor(uniforms.viewMode, vertexIndex, clusterId);
    out.position = position;
    return out;
}

fn getColor(mode: u32, triangle: u32, cluster: u32) -> vec3f {
    if (mode == 0) {
        let x: f32 = f32(triangle) + 1;
        return fract(vec3f(x * 0.1443, x * 0.6841, x * 0.7323));
    }
    if (mode == 1) {
        let x: f32 = f32(cluster) + 1;
        return fract(vec3f(x * 0.1443, x * 0.6841, x * 0.7323));
    }
    return vec3f();
}

@fragment fn fs(in: VertexShaderOut) -> @location(0) vec4f {
    if (uniforms.viewMode == 2) {
        return vec4f(normalize(cross(dpdx(in.position), dpdy(in.position))) * 0.5 + 0.5, 1);
    }
    return vec4f(in.color, 1);
}