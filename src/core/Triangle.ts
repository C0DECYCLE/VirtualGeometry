/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { EdgeIdentifier } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert } from "../utilities/utils.js";
import { float, int } from "../utils.type.js";
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
        const adjacent: Set<Triangle> = new Set<Triangle>();
        this.edges.forEach((edge: Edge) =>
            edge.triangles.forEach((triangle: Triangle) =>
                adjacent.add(triangle),
            ),
        );
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

    public clearEdges(): void {
        this.edges.clear();
    }

    public isBorder(): boolean {
        assert(this.edges.size === 3);
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
}
