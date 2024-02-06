/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

alias ClusterId = u32;
alias VertexId = u32;

struct Uniforms {
    viewProjection: mat4x4f,
    viewMode: u32,
    resolution: vec2f
};

struct Vertex {
    position: vec3f,
};

struct VertexShaderOut {
    @builtin(position) position: vec4f,
    @interpolate(flat) @location(0) color: vec3f
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> tasks: array<ClusterId>;
@group(0) @binding(2) var<storage, read> triangles: array<VertexId>;
@group(0) @binding(3) var<storage, read> vertecies: array<Vertex>;

@vertex fn vs(@builtin(vertex_index) invokeId: u32, @builtin(instance_index) taskId: u32) -> VertexShaderOut {
    let clusterId: ClusterId = tasks[taskId];
    let vertexId: VertexId = triangles[clusterId * 128 * 3 + invokeId];
    let vertex: Vertex = vertecies[vertexId];
    var position: vec3f = vertex.position;

    /*
    if (clusterIndex >= 31) {
        position.y -= 2;
        if (clusterIndex >= 31 + 16) {
            position.y -= 2;
            if (clusterIndex >= 31 + 16 + 8) {
                position.y -= 2;
                if (clusterIndex >= 31 + 16 + 8 + 4) {
                    position.y -= 2;
                    if (clusterIndex >= 31 + 16 + 8 + 4 + 2) {
                        position.y -= 2;
                    }
                }
            }
        }
    }
    */
    /*
    if (clusterIndex >= 136) {
        position.y -= 2;
        if (clusterIndex >= 136 + 68) {
            position.y -= 2;
            if (clusterIndex >= 136 + 68 + 34) {
                position.y -= 2;
                if (clusterIndex >= 136 + 68 + 34 + 17) {
                    position.y -= 2;
                }
            }
        }
    }
    */
    /*
    if (clusterIndex >= 543) {
        position.y -= 2;
        if (clusterIndex >= 543 + 272) {
            position.y -= 2;
            if (clusterIndex >= 543 + 272 + 136) {
                position.y -= 2;
                if (clusterIndex >= 543 + 272 + 136 + 68) {
                    position.y -= 2;
                }
            }
        }
    }
    */
    /*
    if (clusterIndex >= 681) {
        position.y -= 4;
    }
    */


    var out: VertexShaderOut;
    out.position = uniforms.viewProjection * vec4f(position, 1);
    out.color = getColor(uniforms.viewMode, invokeId, clusterId);
    return out;
}

fn getColor(mode: u32, triangle: u32, cluster: u32) -> vec3f {
    if (mode == 0) {
        let x: f32 = f32(triangle) + 1;
        return fract(vec3f(x * 0.1443, x * 0.6841, x * 0.7323));
    }
    if (mode == 1) {
        let x: f32 = f32(cluster) + 1;
        /*
        if (batchIndex >= 543 + 272 + 136 + 68) {
            uid -= 1 - f32(batchIndex % 2);
        }
        */
        return fract(vec3f(x * 0.1443, x * 0.6841, x * 0.7323));
    }
    return vec3f();
}

@fragment fn fs(in: VertexShaderOut) -> @location(0) vec4f {
    return vec4f(in.color, 1);
}