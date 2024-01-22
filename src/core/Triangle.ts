/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { TriangleId } from "../core.type.js";
import { assert } from "../utilities/utils.js";
import { Vertex } from "./Vertex.js";

export class Triangle {
    public readonly globalId: TriangleId;
    public readonly localId: TriangleId;

    public readonly vertices: [Vertex, Vertex, Vertex];
    public readonly adjacent: Set<Triangle>;

    public constructor(
        globalId: TriangleId,
        localId: TriangleId,
        vertices: [Vertex, Vertex, Vertex],
    ) {
        this.globalId = globalId;
        this.localId = localId;
        assert(vertices.length === 3);
        this.vertices = vertices;
        this.adjacent = new Set<Triangle>();
    }
}
