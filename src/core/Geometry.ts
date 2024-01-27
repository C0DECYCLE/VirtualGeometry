/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { OBJParseResult } from "../components/OBJParser.js";
import {
    ClusterSimplifyTryLimit,
    ClusterTrianglesLimit,
    VertexStride,
} from "../constants.js";
import {
    ClusterCenter,
    GeometryData,
    GeometryId,
    GeometryKey,
    VertexId,
} from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { log, warn } from "../utilities/logger.js";
import { simplify } from "../utilities/simplify.js";
import { assert, dotit, swapRemove } from "../utilities/utils.js";
import { Nullable, Undefinable, float, int } from "../utils.type.js";
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
        //this.mergeSimplifyClusters(handlerCount);
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

    private mergeSimplifyClusters(handlerCount: GeometryHandlerCount): void {
        assert(this.clusters);
        /*
        const layer: Cluster[] = [];
        for (let i: int = 0; i < Math.ceil(this.clusters.length / 2); i++) {
            let bases: Cluster[] = [];
            if (this.clusters[i * 2 + 0]) {
                bases.push(this.clusters[i * 2 + 0]);
            }
            if (this.clusters[i * 2 + 1]) {
                bases.push(this.clusters[i * 2 + 1]);
            }
            layer.push(this.mergeSimplify(handlerCount, bases));
        }
        this.clusters.push(...layer);
        */
        this.clusters.push(
            this.mergeSimplify(handlerCount, [this.clusters[1]]),
        );
    }

    private mergeSimplify(
        handlerCount: GeometryHandlerCount,
        clusters: Cluster[],
    ): Cluster {
        assert(clusters.length > 0);
        //let pre: float = performance.now();
        const data: GeometryData = this.merge(clusters);
        //log("merge", dotit(performance.now() - pre), "ms");
        let result: GeometryData = data;
        if (clusters.length > 1) {
            //pre = performance.now();
            result = this.simplify(data, clusters.length);
            //log("simplify", dotit(performance.now() - pre), "ms");
        }
        //pre = performance.now();
        const cluster: Cluster = this.clusterize(handlerCount, result);
        //log("clusterize", dotit(performance.now() - pre), "ms");
        //log(clusters[0], cluster);
        return cluster;
    }

    private merge(clusters: Cluster[]): GeometryData {
        const positions: [float, float, float][] = [];
        const cells: [VertexId, VertexId, VertexId][] = [];
        for (let i: int = 0; i < clusters.length; i++) {
            const triangles: Triangle[] = clusters[i].triangles;
            for (let j: int = 0; j < triangles.length; j++) {
                const index: int = positions.length;
                positions.push(
                    triangles[j].vertices[0].position.toArray(),
                    triangles[j].vertices[1].position.toArray(),
                    triangles[j].vertices[2].position.toArray(),
                );
                cells.push([index + 0, index + 1, index + 2]);
            }
        }
        return this.uniquifyData({ positions, cells });
    }

    private uniquifyData(data: GeometryData): GeometryData {
        const uniques: [float, float, float][] = [];
        for (let i: int = 0; i < data.positions.length; i++) {
            const candidate: [float, float, float] = data.positions[i];
            const duplicate: Nullable<int> = this.findDuplicate(
                uniques,
                candidate,
            );
            if (duplicate === null) {
                this.replaceMatching(data.cells, i, uniques.length);
                uniques.push(candidate);
            } else {
                this.replaceMatching(data.cells, i, duplicate);
            }
        }
        data.positions = uniques;
        return data;
    }

    private findDuplicate(
        uniques: [float, float, float][],
        candidate: [float, float, float],
    ): Nullable<int> {
        for (let i: int = 0; i < uniques.length; i++) {
            const unique: [float, float, float] = uniques[i];
            if (
                candidate[0] === unique[0] &&
                candidate[1] === unique[1] &&
                candidate[2] === unique[2]
            ) {
                return i;
            }
        }
        return null;
    }

    private replaceMatching(
        cells: [VertexId, VertexId, VertexId][],
        match: int,
        replace: int,
    ): void {
        if (match === replace) {
            return;
        }
        for (let i: int = 0; i < cells.length; i++) {
            if (cells[i][0] === match) {
                cells[i][0] = replace;
            }
            if (cells[i][1] === match) {
                cells[i][1] = replace;
            }
            if (cells[i][2] === match) {
                cells[i][2] = replace;
            }
        }
    }

    private simplify(data: GeometryData, divide: int): GeometryData {
        //log(data.positions.length);
        let target: int = Math.ceil(data.positions.length / divide);
        let i: int = 0;
        let result: GeometryData;
        while (true) {
            result = simplify(data)(target);
            let diff: int = result.cells.length - 128;
            //log(target, result, diff);
            if (diff <= 0 && Math.abs(diff) <= 3) {
                break;
            }
            target -= Math.ceil(diff / divide);
            i++;
            if (i > ClusterSimplifyTryLimit) {
                warn("ClusterSimplifyTryLimit hit!");
                break;
            }
        }
        return result;
    }

    private clusterize(
        handlerCount: GeometryHandlerCount,
        data: GeometryData,
    ): Cluster {
        const vertices: Vertex[] = [];
        for (let i: int = 0; i < data.positions.length; i++) {
            vertices.push(
                new Vertex(
                    handlerCount,
                    this.count,
                    undefined,
                    data.positions[i],
                ),
            );
        }
        const triangles: Triangle[] = [];
        for (let i: int = 0; i < data.cells.length; i++) {
            triangles.push(
                new Triangle(handlerCount, this.count, undefined, [
                    vertices[data.cells[i][0]],
                    vertices[data.cells[i][1]],
                    vertices[data.cells[i][2]],
                ]),
            );
        }
        return new Cluster(handlerCount, this.count, triangles, {
            sum: new Vec3(),
            n: 0,
            center: new Vec3(),
        } as ClusterCenter);
    }
}
