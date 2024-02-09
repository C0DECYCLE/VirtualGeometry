/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { EdgeIdentifier, Flushable } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert } from "../utilities/utils.js";
import { float, int } from "../utils.type.js";
import { Count } from "./Count.js";
import { Edge } from "./Edge.js";
import { Vertex } from "./Vertex.js";

export class Triangle {
    public readonly vertices: [Vertex, Vertex, Vertex];

    private edges: Flushable<Map<EdgeIdentifier, Edge>>;

    public constructor(count: Count, vertices: [Vertex, Vertex, Vertex]) {
        assert(vertices.length === 3);
        count.registerTriangle();
        this.vertices = vertices;
        this.edges = new Map<EdgeIdentifier, Edge>();
    }

    public registerEdgesToMap(edges: Map<EdgeIdentifier, Edge>): void {
        this.registerEdgeToMap(edges, this.vertices[0], this.vertices[1]);
        this.registerEdgeToMap(edges, this.vertices[1], this.vertices[2]);
        this.registerEdgeToMap(edges, this.vertices[2], this.vertices[0]);
    }

    private registerEdgeToMap(
        edges: Map<EdgeIdentifier, Edge>,
        a: Vertex,
        b: Vertex,
    ): void {
        const identifier: EdgeIdentifier = Edge.Identify(a, b);
        if (this.hasEdge(identifier)) {
            return;
        }
        if (edges.has(identifier)) {
            const edge: Edge = edges.get(identifier)!;
            edge.triangles.push(this);
            this.registerEdge(edge);
            return;
        }
        const edge: Edge = new Edge([a, b]);
        edge.triangles.push(this);
        this.registerEdge(edge);
        edges.set(edge.identifier, edge);
    }

    public registerEdge(edge: Edge): void {
        assert(this.edges && this.edges.size < 3);
        assert(!this.hasEdge(edge.identifier));
        this.edges.set(edge.identifier, edge);
    }

    public unregisterEdges(bad: [Vertex, Vertex]): EdgeIdentifier[] {
        assert(this.edges && this.edges.size === 3);
        const remove: EdgeIdentifier[] = [];
        this.edges.forEach((edge: Edge) => {
            if (
                edge.vertices.includes(bad[0]) ||
                edge.vertices.includes(bad[1])
            ) {
                this.edges!.delete(edge.identifier);
                remove.push(edge.identifier);
            }
        });
        return remove;
    }

    public hasEdge(identifier: EdgeIdentifier): boolean {
        assert(this.edges);
        return this.edges.has(identifier);
    }

    public clearEdges(): void {
        assert(this.edges);
        this.edges.clear();
    }

    public getAdjacent(): Set<Triangle> {
        assert(this.edges && this.edges.size === 3);
        const adjacent: Set<Triangle> = new Set<Triangle>();
        this.edges.forEach((edge: Edge) =>
            edge.triangles.forEach((triangle: Triangle) =>
                adjacent.add(triangle),
            ),
        );
        adjacent.delete(this);
        return adjacent;
    }

    public isBorder(): boolean {
        assert(this.edges && this.edges.size === 3);
        const edges: Edge[] = Array.from(this.edges.values());
        for (let i: int = 0; i < edges.length; i++) {
            if (edges[i].isBorder()) {
                return true;
            }
        }
        return false;
    }

    public getCenter(): Vec3 {
        return this.vertices[0].position
            .clone()
            .add(this.vertices[1].position)
            .add(this.vertices[2].position)
            .scale(1 / 3);
    }

    public getNormal(): Vec3 {
        const a: Vec3 = this.vertices[1].position
            .clone()
            .sub(this.vertices[0].position);
        const b: Vec3 = this.vertices[2].position
            .clone()
            .sub(this.vertices[0].position);
        return a.cross(b).normalize();
    }

    public getArea(): float {
        const a: Vec3 = this.vertices[1].position
            .clone()
            .sub(this.vertices[0].position);
        const b: Vec3 = this.vertices[2].position
            .clone()
            .sub(this.vertices[0].position);
        return a.cross(b).length() * 0.5;
    }

    public flush(): void {
        assert(this.edges);
        this.edges = null;
    }
}
