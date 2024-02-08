/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterTrianglesLimit } from "../constants.js";
import { ClusterBounds, EdgeIdentifier, VertexId } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert } from "../utilities/utils.js";
import { Nullable, float, int } from "../utils.type.js";
import { Count } from "./Count.js";
import { Edge } from "./Edge.js";
import { GeometryHelper } from "./GeometryHelper.js";
import { Triangle } from "./Triangle.js";
import { Vertex } from "./Vertex.js";

export class Cluster {
    public readonly triangles: Triangle[];
    public readonly bounds: ClusterBounds;
    public readonly border: Set<VertexId>;
    public error: float;
    public parents: Nullable<Cluster[]>;
    public children: Nullable<Cluster[]>;
    public treeChildren: Nullable<Cluster[]>;

    public constructor(count: Count, triangles: Triangle[]) {
        assert(triangles.length <= ClusterTrianglesLimit);
        count.registerCluster();
        this.triangles = triangles;
        this.bounds = this.computeBounds();
        this.border = this.computeBorder();
        this.error = 0;
        this.parents = null;
        this.children = null;
        this.treeChildren = null;
    }

    private computeBounds(): ClusterBounds {
        const min: Vec3 = new Vec3();
        const max: Vec3 = new Vec3();
        for (let i: int = 0; i < this.triangles.length; i++) {
            const triangle: Triangle = this.triangles[i];
            for (let j: int = 0; j < triangle.vertices.length; j++) {
                const vertex: Vertex = triangle.vertices[j];
                min.x = Math.min(min.x, vertex.position.x);
                min.y = Math.min(min.y, vertex.position.y);
                min.z = Math.min(min.z, vertex.position.z);
                max.x = Math.max(max.x, vertex.position.x);
                max.y = Math.max(max.y, vertex.position.y);
                max.z = Math.max(max.z, vertex.position.z);
            }
        }
        return { min, max } as ClusterBounds;
    }

    private computeBorder(): Set<VertexId> {
        const edges: Map<EdgeIdentifier, Edge> = new Map<
            EdgeIdentifier,
            Edge
        >();
        for (let i: int = 0; i < this.triangles.length; i++) {
            const triangle: Triangle = this.triangles[i];
            triangle.clearEdges();
            GeometryHelper.RegisterEdges(edges, triangle);
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

    public getCenter(): Vec3 {
        return this.bounds.min.clone().add(this.bounds.max).scale(0.5);
    }
}
