/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { OBJParseResult } from "../components/OBJParser.js";
import { ClusterLimit } from "../constants.js";
import { ClusterCenter, GeometryId } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { log, warn } from "../utilities/logger.js";
import { assert, dotit, swapRemove } from "../utilities/utils.js";
import { Undefinable, float, int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";
import { Triangle } from "./Triangle.js";
import { Vertex } from "./Vertex.js";

export class Geometry {
    public readonly globalId: GeometryId;

    public readonly clusters: Cluster[];

    public constructor(globalId: GeometryId, parse: OBJParseResult) {
        if (parse.vertexColors) {
            warn("Geometry: Vertex-Colors not supported yet.");
        }
        this.globalId = globalId;
        const pre: float = performance.now();
        this.clusters = this.generateClusters(parse);
        log(
            dotit(this.clusters.length),
            "clusters",
            dotit(performance.now() - pre),
            "ms",
        );
    }

    private generateClusters(parse: OBJParseResult): Cluster[] {
        const vertices: Vertex[] = this.extractVertices(parse);
        const triangles: Triangle[] = this.generateTriangles(parse, vertices);
        this.computeAdjacent(triangles);
        const clusters: Cluster[] = [];
        const unused: Triangle[] = [...triangles];
        let suggestion: Undefinable<Triangle> = undefined;
        while (unused.length !== 0) {
            const group: Triangle[] = [];
            const candidates: Triangle[] = [];
            let center: Undefinable<ClusterCenter> = undefined;
            const first: Triangle = suggestion ? suggestion : unused.pop()!;
            suggestion = undefined;
            center = this.register(first, group, candidates, center);
            while (group.length < ClusterLimit && unused.length !== 0) {
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
            clusters.push(new Cluster(-1, clusters.length, group));
        }
        return clusters;
    }

    private extractVertices(parse: OBJParseResult): Vertex[] {
        const vertices: Vertex[] = [];
        const stride: int = parse.vertices.length / parse.verticesCount;
        for (let i: int = 0; i < parse.verticesCount; i++) {
            const data: Float32Array = parse.vertices.slice(
                i * stride,
                (i + 1) * stride,
            );
            vertices.push(new Vertex(i, -1, data));
        }
        return vertices;
    }

    private generateTriangles(
        parse: OBJParseResult,
        vertices: Vertex[],
    ): Triangle[] {
        assert(parse.indices && parse.indicesCount);
        const triangles: Triangle[] = [];
        const stride: int = 3;
        const count: int = parse.indicesCount / stride;
        for (let i: int = 0; i < count; i++) {
            triangles.push(
                new Triangle(i, -1, [
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
        return { sum: sum, n: n, center: sum.clone().scale(1 / n) };
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
