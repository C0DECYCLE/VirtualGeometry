/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { EdgeIdentifier } from "../core.type.js";
import { assert } from "../utilities/utils.js";
import { Triangle } from "./Triangle.js";
import { Vertex } from "./Vertex.js";

export class Edge {
    public readonly identifier: EdgeIdentifier;
    public readonly vertices: [Vertex, Vertex];
    public readonly triangles: Triangle[];

    public constructor(vertices: [Vertex, Vertex]) {
        assert(vertices.length === 2);
        this.identifier = Edge.Identify(vertices[0], vertices[1]);
        this.vertices = vertices;
        this.triangles = [];
    }

    public static Identify(a: Vertex, b: Vertex): EdgeIdentifier {
        assert(a.id !== b.id);
        return `${a.id < b.id ? a.id : b.id}-${a.id < b.id ? b.id : a.id}`;
    }
}
