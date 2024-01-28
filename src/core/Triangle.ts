/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { EdgeIdentifier } from "../core.type.js";
import { assert } from "../utilities/utils.js";
import { Count } from "./Count.js";
import { Edge } from "./Edge.js";
import { Vertex } from "./Vertex.js";

export class Triangle {
    public readonly vertices: [Vertex, Vertex, Vertex];
    private readonly edges: Map<EdgeIdentifier, Edge>;

    public constructor(count: Count, vertices: [Vertex, Vertex, Vertex]) {
        assert(vertices.length === 3);
        count.registerTriangle();
        this.vertices = vertices;
        this.edges = new Map<EdgeIdentifier, Edge>();
    }

    public registerEdge(edge: Edge): void {
        assert(this.edges.size < 3);
        assert(!this.hasEdge(edge.identifier));
        this.edges.set(edge.identifier, edge);
    }

    public getAdjacent(): Set<Triangle> {
        assert(this.edges.size === 3);
        const triangles: Triangle[] = [];
        this.edges.forEach((edge: Edge) => triangles.push(...edge.triangles));
        const adjacent: Set<Triangle> = new Set<Triangle>(triangles);
        adjacent.delete(this);
        return adjacent;
    }

    public unregisterEdges(bad: [Vertex, Vertex]): EdgeIdentifier[] {
        assert(this.edges.size === 3);
        const remove: EdgeIdentifier[] = [];
        this.edges.forEach((edge: Edge) => {
            if (
                edge.vertices.includes(bad[0]) ||
                edge.vertices.includes(bad[1])
            ) {
                this.edges.delete(edge.identifier);
                remove.push(edge.identifier);
            }
        });
        return remove;
    }

    public hasEdge(identifier: EdgeIdentifier): boolean {
        return this.edges.has(identifier);
    }
}
