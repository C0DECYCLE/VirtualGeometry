/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { OBJParseResult } from "../components/OBJParser.js";
import { ClusterTrianglesLimit, VertexStride } from "../constants.js";
import {
    ClusterCenter,
    EdgeIdentifier,
    GeometryId,
    GeometryKey,
    VertexId,
} from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { log, warn } from "../utilities/logger.js";
import { assert, dotit, swapRemove } from "../utilities/utils.js";
import { Undefinable, float, int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";
import { Count } from "./Count.js";
import { Edge } from "./Edge.js";
import { Triangle } from "./Triangle.js";
import { Vertex } from "./Vertex.js";

export class Geometry {
    public readonly key: GeometryKey;
    public readonly id: GeometryId;

    public readonly vertices: Vertex[];
    public readonly clusters: Cluster[];

    public constructor(count: Count, key: GeometryKey, parse: OBJParseResult) {
        if (parse.vertexColors) {
            warn("Geometry: Vertex-Colors not supported yet.");
        }
        this.key = key;
        this.id = count.registerGeometry();

        const pre: float = performance.now();

        this.vertices = this.extractVertices(count, parse);
        const triangles = this.extractTrianglesEdges(count, parse);
        this.clusters = this.clusterize(count, triangles);
        this.buildHirarchy(count);

        log(
            this.key,
            ":",
            dotit(this.clusters.length),
            "clusters",
            "in",
            dotit(performance.now() - pre),
            "ms",
        );
    }

    private extractVertices(count: Count, parse: OBJParseResult): Vertex[] {
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

    private extractTrianglesEdges(
        count: Count,
        parse: OBJParseResult,
    ): Triangle[] {
        assert(this.vertices && parse.indices && parse.indicesCount);
        const triangles: Triangle[] = [];
        const stride: int = 3;
        const n: int = parse.indicesCount / stride;
        const edges: Map<EdgeIdentifier, Edge> = new Map<
            EdgeIdentifier,
            Edge
        >();
        for (let i: int = 0; i < n; i++) {
            const triangle: Triangle = new Triangle(count, [
                this.vertices[parse.indices[i * stride + 0]],
                this.vertices[parse.indices[i * stride + 1]],
                this.vertices[parse.indices[i * stride + 2]],
            ]);
            triangles.push(triangle);
            this.registerEdges(edges, triangle);
        }
        return triangles;
    }

    private registerEdges(
        edges: Map<EdgeIdentifier, Edge>,
        triangle: Triangle,
    ): void {
        this.registerEdge(
            edges,
            triangle,
            triangle.vertices[0],
            triangle.vertices[1],
        );
        this.registerEdge(
            edges,
            triangle,
            triangle.vertices[1],
            triangle.vertices[2],
        );
        this.registerEdge(
            edges,
            triangle,
            triangle.vertices[2],
            triangle.vertices[0],
        );
    }

    private registerEdge(
        edges: Map<EdgeIdentifier, Edge>,
        triangle: Triangle,
        a: Vertex,
        b: Vertex,
    ): void {
        const identifier: EdgeIdentifier = Edge.Identify(a, b);
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

    private clusterize(count: Count, triangles: Triangle[]): Cluster[] {
        const clusters: Cluster[] = [];
        const unused: Triangle[] = [...triangles];
        let suggestion: Undefinable<Triangle> = undefined;
        while (unused.length !== 0) {
            const group: Triangle[] = [];
            const candidates: Triangle[] = [];
            let center: Undefinable<ClusterCenter> = undefined;
            const first: Triangle = this.popFirst(suggestion, unused);
            suggestion = undefined;
            center = this.register(first, group, candidates, center);
            while (
                group.length < ClusterTrianglesLimit &&
                unused.length !== 0
            ) {
                if (candidates.length === 0) {
                    const { index, nearest } = this.findNearest(center, unused);
                    swapRemove(unused, index);
                    center = this.register(nearest, group, candidates, center);
                    continue;
                }
                const { index, nearest } = this.findNearest(center, candidates);
                swapRemove(candidates, index);
                const nearestInUnusedAt: int = unused.indexOf(nearest);
                if (nearestInUnusedAt === -1) {
                    continue;
                }
                swapRemove(unused, nearestInUnusedAt);
                center = this.register(nearest, group, candidates, center);
            }
            for (let i: int = 0; i < candidates.length; i++) {
                const candidate: Triangle = candidates[i];
                if (!unused.includes(candidate)) {
                    continue;
                }
                suggestion = candidate;
                break;
            }
            clusters.push(new Cluster(count, group /*center*/));
        }
        return clusters;
    }

    private popFirst(
        suggestion: Undefinable<Triangle>,
        unused: Triangle[],
    ): Triangle {
        if (!suggestion) {
            return unused.pop()!;
        }
        const suggestionInUnusedAt: int = unused.indexOf(suggestion);
        assert(suggestionInUnusedAt !== -1);
        swapRemove(unused, suggestionInUnusedAt);
        return suggestion;
    }

    private register(
        triangle: Triangle,
        cluster: Triangle[],
        candidates: Triangle[],
        center?: ClusterCenter,
    ): ClusterCenter {
        cluster.push(triangle);
        candidates.push(...triangle.getAdjacent());
        return this.recomputeCenter(triangle, center);
    }

    private recomputeCenter(
        joining: Triangle,
        base?: ClusterCenter,
    ): ClusterCenter {
        const sum: Vec3 = base !== undefined ? base.sum.clone() : new Vec3();
        sum.add(joining.vertices[0].position);
        sum.add(joining.vertices[1].position);
        sum.add(joining.vertices[2].position);
        const n: int = (base !== undefined ? base.n : 0) + 3;
        return {
            sum: sum,
            n: n,
            center: sum.clone().scale(1 / n),
        } as ClusterCenter;
    }

    private findNearest(
        center: ClusterCenter,
        candidates: Triangle[],
    ): { index: int; nearest: Triangle } {
        let indexNearest: Undefinable<int> = undefined;
        let nearest: Undefinable<Triangle> = undefined;
        let nearestQuadratic: float = Infinity;
        for (let i: int = 0; i < candidates.length; i++) {
            const candidate: Triangle = candidates[i];
            const farestQuadratic: float = Math.max(
                candidate.vertices[0].position
                    .clone()
                    .sub(center.center)
                    .lengthQuadratic(),
                candidate.vertices[1].position
                    .clone()
                    .sub(center.center)
                    .lengthQuadratic(),
                candidate.vertices[2].position
                    .clone()
                    .sub(center.center)
                    .lengthQuadratic(),
            );
            if (farestQuadratic < nearestQuadratic) {
                indexNearest = i;
                nearest = candidate;
                nearestQuadratic = farestQuadratic;
            }
        }
        return { index: indexNearest!, nearest: nearest! };
    }

    private buildHirarchy(count: Count): void {
        this.clusters.push(this.mergeSimplify(count, [this.clusters[1]]));
    }

    private mergeSimplify(count: Count, clusters: Cluster[]): Cluster {
        assert(clusters.length > 0);

        const { triangles, edges } = this.deepMerge(count, clusters);
        edges.sort((a: Edge, b: Edge) => a.lengthQuadratic - b.lengthQuadratic);
        const borderVertices: Set<VertexId> = this.findBorderVertices(edges);

        const collapse: Edge = edges.shift()!;
        assert(!collapse.isBorder());

        const replacement: Vertex = this.createReplacement(
            count,
            collapse,
            borderVertices,
        );

        const remove: int[] = [];
        for (let i: int = 0; i < triangles.length; i++) {
            const triangle: Triangle = triangles[i];
            if (collapse.triangles.includes(triangle)) {
                remove.push(i);
                continue;
            }
            const aInTriangle = triangle.vertices.indexOf(collapse.vertices[0]);
            if (aInTriangle !== -1) {
                triangle.vertices[aInTriangle] = replacement;
            }
            const bInTriangle = triangle.vertices.indexOf(collapse.vertices[1]);
            if (bInTriangle !== -1) {
                triangle.vertices[bInTriangle] = replacement;
            }
        }

        log(remove);
        for (let i: int = remove.length - 1; i >= 0; i--) {
            triangles.splice(remove[i], 1);
        }
        //update the vertices, edges, triangles

        //delete the two collapsed triangles from the triangles repo,
        //from their edges which werent the collapsed one
        //delete the collapsed edge from the edges repo

        //collapse smallest first until there are <= 128 triangles left
        return new Cluster(count, triangles);
    }

    private deepMerge(
        count: Count,
        clusters: Cluster[],
    ): { triangles: Triangle[]; edges: Edge[] } {
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
                this.registerEdges(edges, triangle);
            }
        }
        return { triangles, edges: Array.from(edges.values()) };
    }

    private findBorderVertices(edges: Edge[]): Set<VertexId> {
        const vertices: Set<VertexId> = new Set<VertexId>();
        for (let i: int = 0; i < edges.length; i++) {
            const edge: Edge = edges[i];
            if (edges[i].isBorder()) {
                vertices.add(edge.vertices[0].id);
                vertices.add(edge.vertices[1].id);
            }
        }
        return vertices;
    }

    private createReplacement(
        count: Count,
        edge: Edge,
        border: Set<VertexId>,
    ): Vertex {
        if (border.has(edge.vertices[0].id)) {
            return edge.vertices[0];
        } else if (border.has(edge.vertices[1].id)) {
            return edge.vertices[1];
        }
        const position: Vec3 = new Vec3()
            .copy(edge.vertices[0].position)
            .add(edge.vertices[1].position)
            .scale(0.5);
        const vertex: Vertex = new Vertex(count, position.toArray());
        this.vertices.push(vertex);
        return vertex;
    }
}
