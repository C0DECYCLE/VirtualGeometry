/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { EdgeIdentifier, VertexId } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert } from "../utilities/utils.js";
import { Nullable, float, int } from "../utils.type.js";
import { Count } from "./Count.js";
import { Edge } from "./Edge.js";
import { GeometryHelper } from "./GeometryHelper.js";
import { Triangle } from "./Triangle.js";
import { Vertex } from "./Vertex.js";

export class GeometrySimplify {
    public static Simplify(
        count: Count,
        vertcies: Vertex[],
        triangles: Triangle[],
        edges: Map<EdgeIdentifier, Edge>,
    ): void {
        const self = GeometrySimplify;
        const border: Set<VertexId> = self.FindBorderVertices(edges);
        const reduce: int = Math.ceil(triangles.length / 2);
        while (triangles.length > reduce) {
            try {
                // prettier-ignore
                self.CollapseSmallest(count, vertcies, triangles, edges, border);
            } catch (e) {
                throw new Error(
                    "collapse stopped at " + triangles.length + "/" + reduce,
                );
            }
        }
    }

    private static FindBorderVertices(
        edges: Map<EdgeIdentifier, Edge>,
    ): Set<VertexId> {
        const vertices: Set<VertexId> = new Set<VertexId>();
        edges.forEach((edge: Edge) => {
            if (edge.isBorder()) {
                vertices.add(edge.vertices[0].id);
                vertices.add(edge.vertices[1].id);
            }
        });
        return vertices;
    }

    private static CollapseSmallest(
        count: Count,
        vertices: Vertex[],
        triangles: Triangle[],
        edges: Map<EdgeIdentifier, Edge>,
        border: Set<VertexId>,
    ): void {
        const collapse: Edge = GeometrySimplify.GetNextCollapse(edges, border);
        const replacement: Vertex = GeometrySimplify.CreateReplacement(
            count,
            collapse,
            border,
        );
        vertices.push(replacement);
        const remove: int[] = [];
        for (let i: int = 0; i < triangles.length; i++) {
            const triangle: Triangle = triangles[i];
            const aInTriangle = triangle.vertices.indexOf(collapse.vertices[0]);
            const bInTriangle = triangle.vertices.indexOf(collapse.vertices[1]);
            if (
                collapse.triangles.includes(triangle) ||
                (aInTriangle !== -1 && bInTriangle !== -1)
            ) {
                GeometrySimplify.DeleteBadEdges(triangle, collapse, edges);
                remove.push(i);
                continue;
            }
            if (aInTriangle !== -1) {
                GeometrySimplify.DeleteBadEdges(triangle, collapse, edges);
                triangle.vertices[aInTriangle] = replacement;
                GeometryHelper.RegisterEdges(edges, triangle);
                continue;
            }
            if (bInTriangle !== -1) {
                GeometrySimplify.DeleteBadEdges(triangle, collapse, edges);
                triangle.vertices[bInTriangle] = replacement;
                GeometryHelper.RegisterEdges(edges, triangle);
                continue;
            }
        }
        for (let i: int = remove.length - 1; i >= 0; i--) {
            triangles.splice(remove[i], 1);
        }
    }

    private static GetNextCollapse(
        edges: Map<EdgeIdentifier, Edge>,
        border: Set<VertexId>,
    ): Edge {
        const list: Edge[] = Array.from(edges.values());
        assert(list.length > 0);
        let shortest: Nullable<Edge> = null;
        for (let i: int = 0; i < list.length; i++) {
            const edge: Edge = list[i];
            if (
                edge.isBorder() ||
                (border.has(edge.vertices[0].id) &&
                    border.has(edge.vertices[1].id))
            ) {
                continue;
            }
            if (!shortest) {
                shortest = edge;
                continue;
            }
            if (edge.lengthQuadratic < shortest.lengthQuadratic) {
                shortest = edge;
            }
        }
        assert(shortest);
        return shortest;
    }

    private static CreateReplacement(
        count: Count,
        edge: Edge,
        border: Set<VertexId>,
    ): Vertex {
        if (border.has(edge.vertices[0].id)) {
            return edge.vertices[0];
        } else if (border.has(edge.vertices[1].id)) {
            return edge.vertices[1];
        }

        const a: Vec3 = edge.triangles[0].getNormal();
        const b: Vec3 = edge.triangles[1].getNormal();
        const dot: float = a.dot(b);
        const length: float = Math.sqrt(edge.lengthQuadratic);
        const normal: Vec3 = a.add(b).scale(0.5);

        const position: Vec3 = new Vec3()
            .copy(edge.vertices[0].position)
            .add(edge.vertices[1].position)
            .scale(0.5)
            .add(normal.scale(dot * length * 0.25));
        return new Vertex(count, position.toArray());
    }

    private static DeleteBadEdges(
        triangle: Triangle,
        edge: Edge,
        edges: Map<EdgeIdentifier, Edge>,
    ): void {
        const bad: EdgeIdentifier[] = triangle.unregisterEdges(edge.vertices);
        for (let i: int = 0; i < bad.length; i++) {
            edges.delete(bad[i]);
        }
    }
}
