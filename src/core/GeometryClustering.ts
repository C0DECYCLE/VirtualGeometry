/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { ClusterTrianglesLimit } from "../constants.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert, swapRemove } from "../utilities/utils.js";
import { Nullable, Undefinable, float, int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";
import { Count } from "./Count.js";
import { Triangle } from "./Triangle.js";

export class GeometryClustering {
    public static Clusterize(count: Count, triangles: Triangle[]): Cluster[] {
        const self = GeometryClustering;
        const clusters: Cluster[] = [];
        const unused: Triangle[] = [...triangles];
        let suggestion: Undefinable<Triangle> = undefined;
        const center: Vec3 = self.ComputeCenter(triangles);
        while (unused.length > 0) {
            const use: Triangle[] = [];
            const candidates: Triangle[] = [];
            const first: Triangle = self.PopFirst(suggestion, unused, center);
            suggestion = undefined;
            use.push(first);
            candidates.push(...first.getAdjacent());
            while (use.length < ClusterTrianglesLimit && unused.length > 0) {
                const target: Vec3 = first.getCenter();
                if (candidates.length === 0) {
                    const { index, nearest } = self.FindNearest(target, unused);
                    swapRemove(unused, index);
                    use.push(nearest);
                    candidates.push(...nearest.getAdjacent());
                    continue;
                }
                const { index, nearest } = self.FindNearest(target, candidates);
                swapRemove(candidates, index);
                const nearestInUnusedAt: int = unused.indexOf(nearest);
                if (nearestInUnusedAt !== -1) {
                    swapRemove(unused, nearestInUnusedAt);
                    use.push(nearest);
                    candidates.push(...nearest.getAdjacent());
                }
            }
            clusters.push(new Cluster(count, use));
            const possible: Triangle[] = candidates.filter(
                (candidate: Triangle) => unused.includes(candidate),
            );
            if (possible.length > 0) {
                suggestion = self.FindNearest(
                    clusters[0].getCenter(),
                    possible,
                ).nearest;
            }
        }
        return clusters;
    }

    private static ComputeCenter(triangles: Triangle[]): Vec3 {
        const center: Vec3 = new Vec3();
        for (let i: int = 0; i < triangles.length; i++) {
            const triangle: Triangle = triangles[i];
            center.add(triangle.vertices[0].position);
            center.add(triangle.vertices[1].position);
            center.add(triangle.vertices[2].position);
        }
        center.scale(1 / (triangles.length * 3));
        return center;
    }

    private static PopFirst(
        suggestion: Undefinable<Triangle>,
        unused: Triangle[],
        center: Vec3,
    ): Triangle {
        if (suggestion) {
            const suggestionInUnusedAt: int = unused.indexOf(suggestion);
            assert(suggestionInUnusedAt !== -1);
            swapRemove(unused, suggestionInUnusedAt);
            return suggestion;
        }
        let index: Undefinable<int> = undefined;
        let farest: Nullable<Triangle> = null;
        let distance: float = -1;
        for (let i: int = 0; i < unused.length; i++) {
            const triangle: Triangle = unused[i];
            if (triangle.isBorder()) {
                const dist: float = triangle
                    .getCenter()
                    .sub(center)
                    .lengthQuadratic();
                if (dist > distance) {
                    index = i;
                    farest = triangle;
                    distance = dist;
                }
            }
        }
        if (farest) {
            assert(index !== undefined);
            swapRemove(unused, index);
            return farest;
        }
        return unused.pop()!;
    }

    private static FindNearest(
        target: Vec3,
        candidates: Triangle[],
    ): { index: int; nearest: Triangle } {
        let index: Undefinable<int> = undefined;
        let nearest: Undefinable<Triangle> = undefined;
        let near: float = Infinity;
        for (let i: int = 0; i < candidates.length; i++) {
            const tri: Triangle = candidates[i];
            const dist: float = Math.max(
                tri.vertices[0].position.clone().sub(target).lengthQuadratic(),
                tri.vertices[1].position.clone().sub(target).lengthQuadratic(),
                tri.vertices[2].position.clone().sub(target).lengthQuadratic(),
            );
            if (dist < near) {
                index = i;
                nearest = tri;
                near = dist;
            }
        }
        assert(index !== undefined && nearest);
        return { index, nearest };
    }

    public static ClusterizeWithoutAdjacency(
        count: Count,
        triangles: Triangle[],
    ): Cluster[] {
        const self = GeometryClustering;
        const clusters: Cluster[] = [];
        const unused: Triangle[] = [...triangles];
        let suggestion: Undefinable<Triangle> = undefined;
        const center: Vec3 = self.ComputeCenter(triangles);
        while (unused.length > 0) {
            const use: Triangle[] = [];
            const first: Triangle = self.PopFirst(suggestion, unused, center);
            suggestion = undefined;
            use.push(first);
            while (use.length < ClusterTrianglesLimit && unused.length > 0) {
                const target: Vec3 = first.getCenter();
                const { index, nearest } = self.FindNearest(target, unused);
                swapRemove(unused, index);
                use.push(nearest);
            }
            clusters.push(new Cluster(count, use));
            if (unused.length > 0) {
                suggestion = self.FindNearest(
                    clusters[0].getCenter(),
                    unused,
                ).nearest;
            }
        }
        return clusters;
    }
}
