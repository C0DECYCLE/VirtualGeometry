/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { EdgeIdentifier } from "../core.type.js";
import { assert } from "../utilities/utils.js";
import { float } from "../utils.type.js";
import { Triangle } from "./Triangle.js";
import { Vertex } from "./Vertex.js";

export class Edge {
    public readonly identifier: EdgeIdentifier;
    public readonly vertices: [Vertex, Vertex];
    public readonly lengthQuadratic: float;
    public readonly triangles: Triangle[];

    public constructor(vertices: [Vertex, Vertex]) {
        assert(vertices.length === 2);
        this.identifier = Edge.Identify(vertices[0], vertices[1]);
        this.vertices = vertices;
        this.lengthQuadratic = Edge.LengthQuadratic(vertices[0], vertices[1]);
        this.triangles = [];
    }

    public isBorder(): boolean {
        assert(this.triangles.length > 0);
        return this.triangles.length === 1;
    }

    public static Identify(a: Vertex, b: Vertex): EdgeIdentifier {
        assert(a.id !== b.id);
        return `${a.id < b.id ? a.id : b.id}-${a.id < b.id ? b.id : a.id}`;
    }

    public static LengthQuadratic(a: Vertex, b: Vertex): float {
        return a.position.clone().sub(b.position).lengthQuadratic();
    }
}
