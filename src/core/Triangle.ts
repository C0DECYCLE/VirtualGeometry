/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { assert } from "../utilities/utils.js";
import { Count } from "./Count.js";
import { Edge } from "./Edge.js";
import { Vertex } from "./Vertex.js";

export class Triangle {
    public readonly vertices: [Vertex, Vertex, Vertex];
    private readonly edges: Edge[];

    public constructor(count: Count, vertices: [Vertex, Vertex, Vertex]) {
        assert(vertices.length === 3);
        count.registerTriangle();
        this.vertices = vertices;
        this.edges = [];
    }

    public registerEdge(edge: Edge): void {
        assert(this.edges.length < 3);
        this.edges.push(edge);
    }

    public getAdjacent(): Set<Triangle> {
        assert(this.edges.length === 3);
        const adjacent: Set<Triangle> = new Set<Triangle>([
            ...this.edges[0].triangles,
            ...this.edges[1].triangles,
            ...this.edges[2].triangles,
        ]);
        adjacent.delete(this);
        return adjacent;
    }
}
