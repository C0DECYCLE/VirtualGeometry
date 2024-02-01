/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterTrianglesLimit } from "../constants.js";
import { ClusterBounds, EdgeIdentifier, VertexId } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert } from "../utilities/utils.js";
import { float, int } from "../utils.type.js";
import { Count } from "./Count.js";
import { Edge } from "./Edge.js";
import { GeometryHelper } from "./GeometryHelper.js";
import { Triangle } from "./Triangle.js";

export class Cluster {
    public readonly triangles: Triangle[];
    public readonly bounds: ClusterBounds;
    public readonly border: Set<VertexId>;

    public constructor(count: Count, triangles: Triangle[], center: Vec3) {
        assert(triangles.length <= ClusterTrianglesLimit);
        count.registerCluster();
        this.triangles = triangles;
        this.bounds = this.computeBounds(center);
        this.border = this.computeBorder();
    }

    private computeBounds(center: Vec3): ClusterBounds {
        let radiusQuadratic: float = 0;
        for (let i: int = 0; i < this.triangles.length; i++) {
            const tri: Triangle = this.triangles[i];
            radiusQuadratic = Math.max(
                radiusQuadratic,
                tri.vertices[0].position.clone().sub(center).lengthQuadratic(),
                tri.vertices[1].position.clone().sub(center).lengthQuadratic(),
                tri.vertices[2].position.clone().sub(center).lengthQuadratic(),
            );
        }
        return {
            center: center,
            radiusQuadratic: radiusQuadratic,
        } as ClusterBounds;
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
}
