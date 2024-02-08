/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { OBJParseResult, OBJParser } from "../components/OBJParser.js";
import { Bytes4, ClusterLayout, VertexStride } from "../constants.js";
import { ClusterId, GeometryKey } from "../core.type.js";
import { Nullable, int } from "../utils.type.js";
import { Cluster } from "../core/Cluster.js";
import { Count } from "../core/Count.js";
import { Geometry } from "../core/Geometry.js";
import { Triangle } from "../core/Triangle.js";
import { Vertex } from "../core/Vertex.js";
import { assert } from "../utilities/utils.js";

export class GeometryHandler {
    public readonly count: Count;
    public readonly geometries: Geometry[];

    public clustersBuffer: Nullable<GPUBuffer>;
    public trianglesBuffer: Nullable<GPUBuffer>;
    public verticesBuffer: Nullable<GPUBuffer>;

    public constructor() {
        this.count = new Count(0, 0, 0, 0);
        this.geometries = [];
        this.clustersBuffer = null;
        this.trianglesBuffer = null;
        this.verticesBuffer = null;
    }

    public async import(key: GeometryKey, path: string): Promise<void> {
        const parse: OBJParseResult = OBJParser.Standard.parse(
            await this.loadText(path),
            true,
        );
        //log(await this.existText(path + ".virtual"));
        this.geometries.push(new Geometry(this.count, key, parse));
    }

    private async loadText(path: string): Promise<string> {
        return await fetch(path).then(
            async (response: Response) => await response.text(),
        );
    }

    /*
    private async existText(path: string): Promise<boolean> {
        return await fetch(path, { method: "HEAD" }).then(
            async (response: Response) => response.ok,
        );
    }
    */

    public prepare(device: GPUDevice): void {
        this.prepareBuffers(device);
    }

    private prepareBuffers(device: GPUDevice): void {
        const clusters: ArrayBuffer = this.constructClusters();
        this.clustersBuffer = device.createBuffer({
            label: "general-clusters-buffer",
            size: clusters.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(this.clustersBuffer, 0, clusters);
        const { vertices, triangles } = this.constructTrianglesVertices();
        this.trianglesBuffer = device.createBuffer({
            label: "general-triangles-buffer",
            size: triangles.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(this.trianglesBuffer, 0, triangles);
        this.verticesBuffer = device.createBuffer({
            label: "general-vertices-buffer",
            size: vertices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(this.verticesBuffer, 0, vertices);
    }

    private constructClusters(): ArrayBuffer {
        const arrayBuffer: ArrayBuffer = new ArrayBuffer(
            this.count.clusters * ClusterLayout * Bytes4,
        );
        const floatData: Float32Array = new Float32Array(arrayBuffer);
        const uIntData: Uint32Array = new Uint32Array(arrayBuffer);
        const clusters: Map<Cluster, ClusterId> = new Map<Cluster, ClusterId>();
        for (let i: int = 0; i < this.geometries.length; i++) {
            const geometry: Geometry = this.geometries[i];
            for (let j: int = 0; j < geometry.clusters.length; j++) {
                const id: ClusterId = clusters.size;
                const cluster: Cluster = geometry.clusters[j];
                clusters.set(cluster, id);
            }
        }
        clusters.forEach((id: ClusterId, cluster: Cluster) => {
            floatData[id * ClusterLayout + 0] = cluster.error;
            if (cluster.parents) {
                floatData[id * ClusterLayout + 1] = cluster.parents[0].error;
            }
            const treeChildrenLength: int = cluster.treeChildren
                ? cluster.treeChildren.length
                : 0;
            assert(treeChildrenLength <= 2);
            uIntData[id * ClusterLayout + 2] = treeChildrenLength;
            for (let i: int = 0; i < treeChildrenLength; i++) {
                uIntData[id * ClusterLayout + 3 + i] = clusters.get(
                    cluster.treeChildren![i],
                )!;
            }
        });
        return arrayBuffer;
    }

    private constructTrianglesVertices(): {
        triangles: Uint32Array;
        vertices: Float32Array;
    } {
        const triangles: Uint32Array = new Uint32Array(
            this.count.clusters * 128 * 3,
        );
        const vertices: Float32Array = new Float32Array(
            this.count.vertices * VertexStride,
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
                        triangles[clusterIndex * 128 * 3 + k * 3 + l] =
                            vertex.id;
                    }
                }
                clusterIndex++;
            }
            for (let j: int = 0; j < geometry.vertices.length; j++) {
                const vertex: Vertex = geometry.vertices[j];
                vertex.position.store(vertices, vertex.id * VertexStride);
            }
        }
        return { triangles, vertices };
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
