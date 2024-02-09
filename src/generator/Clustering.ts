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

export class Clustering {
    public static ClusterizeWithAdjacency(
        count: Count,
        triangles: Triangle[],
    ): Cluster[] {
        const clusters: Cluster[] = [];
        const unused: Triangle[] = [...triangles];
        let suggestion: Undefinable<Triangle> = undefined;
        const center: Vec3 = Cluster.ComputeCenter(triangles);
        while (unused.length > 0) {
            const use: Triangle[] = [];
            const candidates: Triangle[] = [];
            // prettier-ignore
            const first: Triangle = Clustering.PopFirst(suggestion, unused, center);
            suggestion = undefined;
            use.push(first);
            candidates.push(...first.getAdjacent());
            while (use.length < ClusterTrianglesLimit && unused.length > 0) {
                const target: Vec3 = first.getCenter();
                if (candidates.length === 0) {
                    // prettier-ignore
                    const { index, nearest } = Clustering.FindNearest(target, unused);
                    swapRemove(unused, index);
                    use.push(nearest);
                    candidates.push(...nearest.getAdjacent());
                    continue;
                }
                // prettier-ignore
                const { index, nearest } = Clustering.FindNearest(target, candidates);
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
                assert(clusters[0].center);
                // prettier-ignore
                suggestion = Clustering.FindNearest(clusters[0].center, possible).nearest;
            }
        }
        return clusters;
    }

    public static ClusterizeWithoutAdjacency(
        count: Count,
        triangles: Triangle[],
    ): Cluster[] {
        const clusters: Cluster[] = [];
        const unused: Triangle[] = [...triangles];
        let suggestion: Undefinable<Triangle> = undefined;
        const center: Vec3 = Cluster.ComputeCenter(triangles);
        while (unused.length > 0) {
            const use: Triangle[] = [];
            // prettier-ignore
            const first: Triangle = Clustering.PopFirst(suggestion, unused, center);
            suggestion = undefined;
            use.push(first);
            while (use.length < ClusterTrianglesLimit && unused.length > 0) {
                const target: Vec3 = first.getCenter();
                // prettier-ignore
                const { index, nearest } = Clustering.FindNearest(target, unused);
                swapRemove(unused, index);
                use.push(nearest);
            }
            clusters.push(new Cluster(count, use));
            if (unused.length > 0) {
                assert(clusters[0].center);
                // prettier-ignore
                suggestion = Clustering.FindNearest(clusters[0].center, unused).nearest;
            }
        }
        return clusters;
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
}
