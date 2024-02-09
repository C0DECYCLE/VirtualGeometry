/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { Count } from "./Count.js";
import { Geometry } from "../components/Geometry.js";
import {
    Bytes4,
    ClusterLayout,
    ClusterTrianglesLimit,
    VertexLayout,
} from "../constants.js";
import { ClusterId, Flushable, GeometryKey } from "../core.type.js";
import { assert, clear } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";
import { Virtual } from "./Virtual.js";
import { Triangle } from "./Triangle.js";
import { Vertex } from "./Vertex.js";

export class Generator {
    private static readonly Count: Count = new Count(0, 0, 0, 0);
    private static Cache: Flushable<Virtual[]> = [];

    public static async Generate(
        key: GeometryKey,
        path: string,
    ): Promise<Geometry> {
        assert(Generator.Cache);
        const virtual: Virtual = await Virtual.Create(
            Generator.Count,
            key,
            path,
        );
        Generator.Cache.push(virtual);
        return Geometry.FromVirtual(virtual);
    }

    public static Compact(device: GPUDevice): {
        clusters: GPUBuffer;
        triangles: GPUBuffer;
        vertices: GPUBuffer;
        rootIds: Map<GeometryKey, ClusterId>;
    } {
        const rootIds: Map<GeometryKey, ClusterId> = new Map<
            GeometryKey,
            ClusterId
        >();
        const clusters: ArrayBuffer = Generator.CompactClusters(rootIds);
        const { triangles, vertices } = Generator.CompactTrianglesVertices();
        Generator.Flush();
        return {
            clusters: Generator.Buffer("clusters", clusters, device),
            triangles: Generator.Buffer("triangles", triangles, device),
            vertices: Generator.Buffer("vertices", vertices, device),
            rootIds,
        };
    }

    private static Flush(): void {
        assert(Generator.Cache);
        clear(Generator.Cache);
        Generator.Cache = null;
    }

    private static Buffer(
        label: string,
        data: ArrayBuffer,
        device: GPUDevice,
    ): GPUBuffer {
        const buffer: GPUBuffer = device.createBuffer({
            label: `${label}-buffer`,
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    }

    private static CompactClusters(
        rootIds: Map<GeometryKey, ClusterId>,
    ): ArrayBuffer {
        const bytes: int = Generator.Count.clusters * ClusterLayout * Bytes4;
        const arrayBuffer: ArrayBuffer = new ArrayBuffer(bytes);
        const floatData: Float32Array = new Float32Array(arrayBuffer);
        const uIntData: Uint32Array = new Uint32Array(arrayBuffer);
        const clusters: Map<Cluster, ClusterId> =
            Generator.MapClusters(rootIds);
        Generator.StoreClusters(clusters, floatData, uIntData);
        return arrayBuffer;
    }

    private static MapClusters(
        rootIds: Map<GeometryKey, ClusterId>,
    ): Map<Cluster, ClusterId> {
        assert(Generator.Cache);
        const clusters: Map<Cluster, ClusterId> = new Map<Cluster, ClusterId>();
        for (let i: int = 0; i < Generator.Cache.length; i++) {
            const virtual: Virtual = Generator.Cache[i];
            for (let j: int = 0; j < virtual.clusters.length; j++) {
                const id: ClusterId = clusters.size;
                const cluster: Cluster = virtual.clusters[j];
                clusters.set(cluster, id);
            }
            const root: Cluster = virtual.clusters[virtual.clusters.length - 1];
            assert(!rootIds.has(virtual.key));
            rootIds.set(virtual.key, clusters.get(root)!);
            assert(rootIds.get(virtual.key) !== undefined);
        }
        return clusters;
    }

    private static StoreClusters(
        clusters: Map<Cluster, ClusterId>,
        floatData: Float32Array,
        uIntData: Uint32Array,
    ): void {
        clusters.forEach((id: ClusterId, cluster: Cluster) => {
            floatData[id * ClusterLayout + 0] = cluster.error;
            floatData[id * ClusterLayout + 1] = cluster.parentError || 0;
            const tree: Nullable<Cluster[]> = cluster.tree;
            const length: int = tree ? tree.length : 0;
            assert(length <= 2);
            uIntData[id * ClusterLayout + 2] = length;
            for (let i: int = 0; i < length; i++) {
                uIntData[id * ClusterLayout + 3 + i] = clusters.get(tree![i])!;
            }
        });
    }

    private static CompactTrianglesVertices(): {
        triangles: Uint32Array;
        vertices: Float32Array;
    } {
        assert(Generator.Cache);
        const trianglesBytes: int =
            Generator.Count.clusters * ClusterTrianglesLimit * 3;
        const verticesBytes: int = Generator.Count.vertices * VertexLayout;
        const triangles: Uint32Array = new Uint32Array(trianglesBytes);
        const vertices: Float32Array = new Float32Array(verticesBytes);
        let index: int = 0;
        for (let i: int = 0; i < Generator.Cache.length; i++) {
            const virtual: Virtual = Generator.Cache[i];
            for (let j: int = 0; j < virtual.clusters.length; j++) {
                const cluster: Cluster = virtual.clusters[j];
                Generator.StoreTriangles(triangles, index, cluster);
                index++;
            }
            Generator.StoreVertices(vertices, virtual);
        }
        return { triangles, vertices };
    }

    private static StoreTriangles(
        triangles: Uint32Array,
        index: int,
        cluster: Cluster,
    ): void {
        for (let k: int = 0; k < cluster.triangles.length; k++) {
            const triangle: Triangle = cluster.triangles[k];
            for (let l: int = 0; l < triangle.vertices.length; l++) {
                const vertex: Vertex = triangle.vertices[l];
                triangles[index * ClusterTrianglesLimit * 3 + k * 3 + l] =
                    vertex.id;
            }
        }
    }

    private static StoreVertices(
        vertices: Float32Array,
        virtual: Virtual,
    ): void {
        for (let j: int = 0; j < virtual.vertices.length; j++) {
            const vertex: Vertex = virtual.vertices[j];
            vertex.position.store(vertices, vertex.id * VertexLayout);
        }
    }
}
