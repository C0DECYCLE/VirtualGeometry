/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterTrianglesLimit } from "../constants.js";
import { EdgeIdentifier, Flushable, VertexId } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert } from "../utilities/utils.js";
import { Nullable, float, int } from "../utils.type.js";
import { Count } from "./Count.js";
import { Edge } from "./Edge.js";
import { Triangle } from "./Triangle.js";

export class Cluster {
    public readonly triangles: Triangle[];
    public error: float;
    public parentError: Nullable<float>;
    public tree: Nullable<Cluster[]>;

    public center: Flushable<Vec3>;
    public border: Flushable<Set<VertexId>>;
    public childrenLength: Flushable<Nullable<int>>;

    public constructor(count: Count, triangles: Triangle[]) {
        assert(triangles.length <= ClusterTrianglesLimit);
        count.registerCluster();
        this.triangles = triangles;
        this.center = Cluster.ComputeCenter(this.triangles);
        this.border = Cluster.ComputeBorder(this.triangles);
        this.error = Cluster.ComputeArea(this.triangles);
        this.parentError = null;
        this.childrenLength = null;
        this.tree = null;
    }

    public flush(): void {
        assert(this.center && this.border);
        this.center = null;
        this.border = null;
        this.childrenLength = null;
        for (let i: int = 0; i < this.triangles.length; i++) {
            this.triangles[i].flush();
        }
    }

    public static ComputeCenter(triangles: Triangle[]): Vec3 {
        const center: Vec3 = new Vec3();
        for (let i: int = 0; i < triangles.length; i++) {
            const triangle: Triangle = triangles[i];
            center.add(triangle.vertices[0].position);
            center.add(triangle.vertices[1].position);
            center.add(triangle.vertices[2].position);
        }
        center.scale(1 / (triangles.length * 3));
        return center;
    }

    public static ComputeBorder(triangles: Triangle[]): Set<VertexId> {
        const edges: Map<EdgeIdentifier, Edge> = new Map<
            EdgeIdentifier,
            Edge
        >();
        for (let i: int = 0; i < triangles.length; i++) {
            const triangle: Triangle = triangles[i];
            triangle.clearEdges();
            triangle.registerEdgesToMap(edges);
        }
        const vertices: Set<VertexId> = new Set<VertexId>();
        edges.forEach((edge: Edge) => {
            if (edge.isBorder()) {
                vertices.add(edge.vertices[0].id);
                vertices.add(edge.vertices[1].id);
            }
        });
        return vertices;
    }

    public static ComputeArea(triangles: Triangle[]): float {
        let sum: float = 0;
        for (let i: int = 0; i < triangles.length; i++) {
            sum += triangles[i].getArea();
        }
        return sum;
    }
}
