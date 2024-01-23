/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { TriangleId } from "../core.type.js";
import { assert } from "../utilities/utils.js";
import { Undefinable } from "../utils.type.js";
import { GeometryHandlerCount, GeometryCount, ClusterCount } from "./Counts.js";
import { Vertex } from "./Vertex.js";

export class Triangle {
    public readonly inHandlerId: TriangleId;
    public readonly inGeometryId: TriangleId;
    public inClusterId: Undefinable<TriangleId>;

    public readonly vertices: [Vertex, Vertex, Vertex];
    public readonly adjacent: Set<Triangle>;

    public constructor(
        handlerCount: GeometryHandlerCount,
        geometryCount: GeometryCount,
        clusterCount: Undefinable<ClusterCount>,
        vertices: [Vertex, Vertex, Vertex],
    ) {
        this.inHandlerId = handlerCount.registerTriangle();
        this.inGeometryId = geometryCount.registerTriangle();
        if (clusterCount) {
            this.inClusterId = clusterCount.registerTriangle();
        }
        assert(vertices.length === 3);
        this.vertices = vertices;
        this.adjacent = new Set<Triangle>();
    }

    public register(clusterCount: ClusterCount): void {
        assert(this.inClusterId === undefined);
        this.inClusterId = clusterCount.registerTriangle();
    }
}
