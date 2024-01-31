/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { OBJParseResult } from "../components/OBJParser.js";
import { VertexStride } from "../constants.js";
import { EdgeIdentifier } from "../core.type.js";
import { assert } from "../utilities/utils.js";
import { int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";
import { Count } from "./Count.js";
import { Edge } from "./Edge.js";
import { Triangle } from "./Triangle.js";
import { Vertex } from "./Vertex.js";

export class GeometryHelper {
    public static ExtractVertices(
        count: Count,
        parse: OBJParseResult,
    ): Vertex[] {
        const vertices: Vertex[] = [];
        const stride: int = parse.vertices.length / parse.verticesCount;
        assert(stride === VertexStride);
        for (let i: int = 0; i < parse.verticesCount; i++) {
            const vertex: Vertex = new Vertex(
                count,
                parse.vertices.slice(i * stride, (i + 1) * stride),
            );
            vertices.push(vertex);
        }
        return vertices;
    }

    public static ExtractTrianglesEdges(
        count: Count,
        parse: OBJParseResult,
        vertices: Vertex[],
    ): Triangle[] {
        assert(parse.indices && parse.indicesCount);
        const triangles: Triangle[] = [];
        const stride: int = 3;
        const n: int = parse.indicesCount / stride;
        const edges: Map<EdgeIdentifier, Edge> = new Map<
            EdgeIdentifier,
            Edge
        >();
        for (let i: int = 0; i < n; i++) {
            const triangle: Triangle = new Triangle(count, [
                vertices[parse.indices[i * stride + 0]],
                vertices[parse.indices[i * stride + 1]],
                vertices[parse.indices[i * stride + 2]],
            ]);
            triangles.push(triangle);
            GeometryHelper.RegisterEdges(edges, triangle);
        }
        return triangles;
    }

    public static MergeClusters(
        count: Count,
        clusters: Cluster[],
    ): { triangles: Triangle[]; edges: Map<EdgeIdentifier, Edge> } {
        assert(clusters.length > 0);
        const triangles: Triangle[] = [];
        const edges: Map<EdgeIdentifier, Edge> = new Map<
            EdgeIdentifier,
            Edge
        >();
        for (let i: int = 0; i < clusters.length; i++) {
            const cluster: Cluster = clusters[i];
            for (let j: int = 0; j < cluster.triangles.length; j++) {
                const triangle: Triangle = new Triangle(count, [
                    ...cluster.triangles[j].vertices,
                ]);
                triangles.push(triangle);
                GeometryHelper.RegisterEdges(edges, triangle);
            }
        }
        return { triangles, edges };
    }

    public static RegisterEdges(
        edges: Map<EdgeIdentifier, Edge>,
        triangle: Triangle,
    ): void {
        GeometryHelper.RegisterEdge(
            edges,
            triangle,
            triangle.vertices[0],
            triangle.vertices[1],
        );
        GeometryHelper.RegisterEdge(
            edges,
            triangle,
            triangle.vertices[1],
            triangle.vertices[2],
        );
        GeometryHelper.RegisterEdge(
            edges,
            triangle,
            triangle.vertices[2],
            triangle.vertices[0],
        );
    }

    private static RegisterEdge(
        edges: Map<EdgeIdentifier, Edge>,
        triangle: Triangle,
        a: Vertex,
        b: Vertex,
    ): void {
        const identifier: EdgeIdentifier = Edge.Identify(a, b);
        if (triangle.hasEdge(identifier)) {
            return;
        }
        if (edges.has(identifier)) {
            const edge: Edge = edges.get(identifier)!;
            edge.triangles.push(triangle);
            triangle.registerEdge(edge);
            return;
        }
        const edge: Edge = new Edge([a, b]);
        edge.triangles.push(triangle);
        triangle.registerEdge(edge);
        edges.set(edge.identifier, edge);
    }
}
