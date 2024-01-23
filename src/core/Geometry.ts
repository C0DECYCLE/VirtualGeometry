/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { OBJParseResult } from "../components/OBJParser.js";
import { ClusterTrianglesLimit, VertexStride } from "../constants.js";
import { ClusterCenter, GeometryId, GeometryKey } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { log, warn } from "../utilities/logger.js";
import { assert, dotit, swapRemove } from "../utilities/utils.js";
import { Undefinable, float, int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";
import { GeometryCount, GeometryHandlerCount } from "./Counts.js";
import { Triangle } from "./Triangle.js";
import { Vertex } from "./Vertex.js";

export class Geometry {
    public readonly key: GeometryKey;
    public readonly inHandlerId: GeometryId;
    public readonly count: GeometryCount;

    public readonly clusters: Cluster[];

    public constructor(
        key: GeometryKey,
        handlerCount: GeometryHandlerCount,
        parse: OBJParseResult,
    ) {
        this.key = key;
        this.inHandlerId = handlerCount.registerGeometry();
        this.count = new GeometryCount(0, 0, 0);
        if (parse.vertexColors) {
            warn("Geometry: Vertex-Colors not supported yet.");
        }

        const pre: float = performance.now();
        this.clusters = this.generateClusters(handlerCount, parse);
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

    private generateClusters(
        handlerCount: GeometryHandlerCount,
        parse: OBJParseResult,
    ): Cluster[] {
        const vertices: Vertex[] = this.extractVertices(handlerCount, parse);
        const triangles: Triangle[] = this.generateTriangles(
            handlerCount,
            parse,
            vertices,
        );
        this.computeAdjacent(triangles);
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
            clusters.push(new Cluster(handlerCount, this.count, group, center));
        }
        return clusters;
    }

    private extractVertices(
        handlerCount: GeometryHandlerCount,
        parse: OBJParseResult,
    ): Vertex[] {
        const vertices: Vertex[] = [];
        const stride: int = parse.vertices.length / parse.verticesCount;
        assert(stride === VertexStride);
        for (let i: int = 0; i < parse.verticesCount; i++) {
            const data: Float32Array = parse.vertices.slice(
                i * stride,
                (i + 1) * stride,
            );
            vertices.push(
                new Vertex(handlerCount, this.count, undefined, data),
            );
        }
        return vertices;
    }

    private generateTriangles(
        handlerCount: GeometryHandlerCount,
        parse: OBJParseResult,
        vertices: Vertex[],
    ): Triangle[] {
        assert(parse.indices && parse.indicesCount);
        const triangles: Triangle[] = [];
        const stride: int = 3;
        const count: int = parse.indicesCount / stride;
        for (let i: int = 0; i < count; i++) {
            triangles.push(
                new Triangle(handlerCount, this.count, undefined, [
                    vertices[parse.indices[i * stride + 0]],
                    vertices[parse.indices[i * stride + 1]],
                    vertices[parse.indices[i * stride + 2]],
                ]),
            );
        }
        return triangles;
    }

    private computeAdjacent(triangles: Triangle[]): void {
        for (let i: int = 0; i < triangles.length; i++) {
            const target: Triangle = triangles[i];
            if (target.adjacent.size > 2) {
                continue;
            }
            for (let j: int = 0; j < triangles.length; j++) {
                if (i === j) {
                    continue;
                }
                const candidate: Triangle = triangles[j];
                let matching: int = 0;
                if (target.vertices.includes(candidate.vertices[0])) {
                    matching++;
                }
                if (target.vertices.includes(candidate.vertices[1])) {
                    matching++;
                }
                if (target.vertices.includes(candidate.vertices[2])) {
                    matching++;
                }
                if (matching > 1) {
                    target.adjacent.add(candidate);
                    candidate.adjacent.add(target);
                }
                if (target.adjacent.size > 2) {
                    break;
                }
            }
        }
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
        candidates.push(...triangle.adjacent);
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
}
