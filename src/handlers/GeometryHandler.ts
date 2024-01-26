/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { OBJParseResult, OBJParser } from "../components/OBJParser.js";
import { VertexStride } from "../constants.js";
import { GeometryKey } from "../core.type.js";
import { Nullable, int } from "../utils.type.js";
import { Cluster } from "../core/Cluster.js";
import { GeometryHandlerCount } from "../core/Counts.js";
import { Geometry } from "../core/Geometry.js";
import { Triangle } from "../core/Triangle.js";
import { Vertex } from "../core/Vertex.js";

export class GeometryHandler {
    public readonly count: GeometryHandlerCount;

    public readonly geometries: Geometry[];

    public verticesBuffer: Nullable<GPUBuffer>;
    public indicesBuffer: Nullable<GPUBuffer>;

    public constructor() {
        this.count = new GeometryHandlerCount(0, 0, 0, 0);
        this.geometries = [];
        this.verticesBuffer = null;
        this.indicesBuffer = null;
    }

    public async import(key: GeometryKey, path: string): Promise<void> {
        const parse: OBJParseResult = OBJParser.Standard.parse(
            await this.loadText(path),
            true,
        );
        this.geometries.push(new Geometry(key, this.count, parse));
    }

    private async loadText(path: string): Promise<string> {
        return await fetch(path).then(
            async (response: Response) => await response.text(),
        );
    }

    public prepare(device: GPUDevice): void {
        this.prepareBuffers(device);
    }

    private prepareBuffers(device: GPUDevice): void {
        const { vertices, indices } = this.constructVerticesIndices();
        this.verticesBuffer = device.createBuffer({
            label: "general-vertices-buffer",
            size: vertices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(this.verticesBuffer, 0, vertices);
        this.indicesBuffer = device.createBuffer({
            label: "general-indices-buffer",
            size: indices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(this.indicesBuffer, 0, indices);
    }

    private constructVerticesIndices(): {
        vertices: Float32Array;
        indices: Uint32Array;
    } {
        const vertices: Float32Array = new Float32Array(
            this.count.vertices * VertexStride,
        );
        const indices: Uint32Array = new Uint32Array(
            this.count.clusters * 128 * 3,
        );
        let clusterIndex: int = 0;
        for (let i: int = 0; i < this.geometries.length; i++) {
            const geometry: Geometry = this.geometries[i];
            for (let j: int = 0; j < geometry.clusters.length; j++) {
                const cluster: Cluster = geometry.clusters[j];
                for (let k: int = 0; k < cluster.triangles.length; k++) {
                    const triangle: Triangle = cluster.triangles[k];
                    for (let l: int = 0; l < triangle.vertices.length; l++) {
                        const vertex: Vertex = triangle.vertices[l];
                        vertex.position.store(
                            vertices,
                            vertex.inHandlerId * VertexStride,
                        );
                        indices[clusterIndex * 128 * 3 + k * 3 + l] =
                            vertex.inHandlerId;
                    }
                }
                clusterIndex++;
            }
        }
        return { vertices, indices };
    }

    public exists(key: GeometryKey): boolean {
        for (let i: int = 0; i < this.geometries.length; i++) {
            if (this.geometries[i].key === key) {
                return true;
            }
        }
        return false;
    }
}
