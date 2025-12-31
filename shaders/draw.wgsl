/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */
 
enable primitive_index;

alias EntityIndex = u32;

alias ClusterId = u32;

alias VertexId = u32;

struct DrawPair {
    entity: EntityIndex,
    cluster: ClusterId
};

struct Uniforms {
    viewProjection: mat4x4f,
    viewMode: u32,
    cameraPosition: vec3f
};

struct Vertex {
    position: vec3f
};

struct Entity {
    position: vec3f,
    root: ClusterId,
    radius: f32
};

struct VertexShaderOut {
    @builtin(position) clipspace: vec4f,
    @interpolate(flat) @location(0) color: vec3f,
    @location(1) position: vec3f
};

override TRIANGLES_LIMIT: u32;
const TRIANGLE_VERTICES: u32 = 3;

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> drawPairs: array<DrawPair>;
@group(0) @binding(2) var<storage, read> entities: array<Entity>;
@group(0) @binding(3) var<storage, read> triangles: array<VertexId>;
@group(0) @binding(4) var<storage, read> vertices: array<Vertex>;

@vertex fn vs(
    @builtin(vertex_index) vertexIndex: u32, 
    @builtin(instance_index) instanceIndex: u32
) -> VertexShaderOut {
    let drawPair: DrawPair = drawPairs[instanceIndex];
    let entity: Entity = entities[drawPair.entity];
    let vertexId: VertexId = triangles[drawPair.cluster * TRIANGLES_LIMIT * TRIANGLE_VERTICES + vertexIndex];
    let vertex: Vertex = vertices[vertexId];
    let position: vec3f = vertex.position + entity.position;
    var out: VertexShaderOut;
    out.clipspace = uniforms.viewProjection * vec4f(position, 1);
    out.color = getColor(uniforms.viewMode, vertexIndex, drawPair.cluster, drawPair.entity);
    out.position = position;
    return out;
}

fn getColor(mode: u32, triangle: u32, cluster: u32, entity: u32) -> vec3f {
    if (mode == 0) {
        let x: f32 = f32(entity + cluster + triangle) + 1;
        return fract(vec3f(x * 0.1443, x * 0.6841, x * 0.7323));
    }
    if (mode == 1) {
        let x: f32 = f32(entity + cluster) + 1;
        return fract(vec3f(x * 0.1443, x * 0.6841, x * 0.7323));
    }
    if (mode == 2) {
        let x: f32 = f32(entity) + 1;
        return fract(vec3f(x * 0.1443, x * 0.6841, x * 0.7323));
    }
    return vec3f();
}

@fragment fn fs(in: VertexShaderOut, @builtin(primitive_index) pid: u32) -> @location(0) vec4f {
    //let x: f32 = f32(pid) + 1;
    //return vec4f(fract(vec3f(x * 0.1443, x * 0.6841, x * 0.7323)), 1);
    if (uniforms.viewMode == 3) {
        return vec4f(normalize(cross(dpdx(in.position), dpdy(in.position))) * 0.5 + 0.5, 1);
    }
    return vec4f(in.color, 1);
    
}