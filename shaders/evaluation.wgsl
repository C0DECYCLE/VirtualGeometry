/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

alias ClusterId = u32;

struct Uniforms {
    viewProjection: mat4x4f,
    viewMode: u32,
    resolution: vec2f
};

struct Cluster {
    min: vec3f,
    max: vec3f,
    parents: array<ClusterId, 2>,
    children: array<ClusterId, 4>,
    error: f32
};

struct Indirect {
    invokeCount: u32,
    taskCount: atomic<u32>,
    firstInvoke: u32,
    firstTask: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read_write> tasks: array<ClusterId>;
@group(0) @binding(3) var<storage, read_write> indirect: Indirect;

@compute @workgroup_size(1, 1, 1) fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
    /*
    let rootId: ClusterId = 14;
    var queue: array<ClusterId, 100> = array<ClusterId, 100>();
    var queueLength: u32 = 0;
    queue[queueLength] = rootId;
    queueLength += 1;
    while(queueLength > 0) {
        let id: ClusterId = queue[queueLength - 1];
        queueLength -= 1;
        let cluster: Cluster = clusters[id];
        let error: f32 = size(cluster);

        
        var parentsLength: u32 = 0;
        if (cluster.parents[0] != 0) {
            parentsLength += 1;
        } 
        if (cluster.parents[1] != 0) {
            parentsLength += 1;
        } 

        var childrenLength: u32 = 0;
        if (cluster.children[0] != 0) {
            childrenLength += 1;
        } 
        if (cluster.children[1] != 0) {
            childrenLength += 1;
        } 
        if (cluster.children[2] != 0) {
            childrenLength += 1;
        } 
        if (cluster.children[3] != 0) {
            childrenLength += 1;
        } 

        if (childrenLength == 0) {
            tasks[atomicAdd(&indirect.taskCount, 1)] = id;
        } else if (error < 0.5) {
            if (parentsLength == 0) {
                tasks[atomicAdd(&indirect.taskCount, 1)] = id;
            
            } else {
                let parent: Cluster = clusters[cluster.parents[0]];
                if (parent.children[0] != 0) {
                    tasks[atomicAdd(&indirect.taskCount, 1)] = parent.children[0];
                } 
                if (parent.children[1] != 0) {
                    tasks[atomicAdd(&indirect.taskCount, 1)] = parent.children[1];
                } 
                if (parent.children[2] != 0) {
                    tasks[atomicAdd(&indirect.taskCount, 1)] = parent.children[2];
                } 
                if (parent.children[3] != 0) {
                    tasks[atomicAdd(&indirect.taskCount, 1)] = parent.children[3];
                } 
            }
            
        } else {
            if (cluster.children[0] != 0) {
                queue[queueLength] = cluster.children[0] - 1;
                queueLength += 1;
            } 
            if (cluster.children[1] != 0) {
                queue[queueLength] = cluster.children[1] - 1;
                queueLength += 1;
            } 
            
            if (cluster.children[2] != 0) {
                queue[queueLength] = cluster.children[2] - 1;
                queueLength += 1;
            } 
            if (cluster.children[3] != 0) {
                queue[queueLength] = cluster.children[3] - 1;
                queueLength += 1;
            } 
            
        }
    }
    */
    /*
    let threshold: f32 = 0.2; //based on distance
    for (var id: u32 = 0; id < 62; id++) {
        let cluster: Cluster = clusters[id];

        var maxParentError: f32 = 0;
        if (cluster.parents[0] != 0) {
            maxParentError = max(maxParentError, error(clusters[cluster.parents[0] - 1]));
        } 
        if (cluster.parents[1] != 0) {
            maxParentError = max(maxParentError, error(clusters[cluster.parents[1] - 1]));
        } 
        let clusterError: f32 = error(cluster);

        //better error calculation offline and monotonic!

        if (maxParentError > threshold && clusterError <= threshold) { //check no children
            tasks[atomicAdd(&indirect.taskCount, 1)] = id;
        }
    }
    */
    let u: vec2f = vec2f(2) + uniforms.resolution;
    let threshold: f32 = 0.5; //based on distance
    for (var id: u32 = 0; id < 62; id++) {
        let cluster: Cluster = clusters[id];

        var parentError: f32 = 0;
        if (cluster.parents[0] != 0) {
            parentError = clusters[cluster.parents[0] - 1].error;
        } 
        if (cluster.parents[1] != 0) {
            parentError = clusters[cluster.parents[1] - 1].error;
        } 
        let clusterError: f32 = cluster.error;

        if (parentError > threshold && clusterError <= threshold) { //check no children
            tasks[atomicAdd(&indirect.taskCount, 1)] = id;
        }
    }
}

fn error(cluster: Cluster) -> f32 {
    //let dir: vec3f = cluster.max - cluster.min;
    let cMin: vec3f = cluster.min;// - dir;
    let cMax: vec3f = cluster.max;// + dir;
    let a: vec2f = project(vec3f(cMin.x, cMin.y, cMin.z));
    let b: vec2f = project(vec3f(cMax.x, cMin.y, cMin.z));
    let c: vec2f = project(vec3f(cMax.x, cMin.y, cMax.z));
    let d: vec2f = project(vec3f(cMin.x, cMin.y, cMax.z));
    let e: vec2f = project(vec3f(cMin.x, cMax.y, cMin.z));
    let f: vec2f = project(vec3f(cMax.x, cMax.y, cMin.z));
    let g: vec2f = project(vec3f(cMax.x, cMax.y, cMax.z));
    let h: vec2f = project(vec3f(cMin.x, cMax.y, cMax.z));
    let sMin: vec2f = vec2f(
        min(min(min(min(min(min(min(a.x, b.x), c.x), d.x), e.x), f.x), g.x), h.x),
        min(min(min(min(min(min(min(a.y, b.y), c.y), d.y), e.y), f.y), g.y), h.y),
    );
    let sMax: vec2f = vec2f(
        max(max(max(max(max(max(max(a.x, b.x), c.x), d.x), e.x), f.x), g.x), h.x),
        max(max(max(max(max(max(max(a.y, b.y), c.y), d.y), e.y), f.y), g.y), h.y),
    );
    return length(sMax - sMin);
}

fn project(position: vec3f) -> vec2f {
    let viewspace: vec4f = uniforms.viewProjection * vec4f(position, 1.0);
    var clipspace: vec3f = viewspace.xyz / viewspace.w;
    clipspace.x = (clipspace.x / 2 + 0.5) + 0 * uniforms.resolution.x;
    clipspace.y = (clipspace.y / 2 + 0.5) + 0 * uniforms.resolution.y;
    return clipspace.xy;
}