/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterTrianglesLimit } from "../constants.js";
import { ClusterCenter } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert, swapRemove } from "../utilities/utils.js";
import { Undefinable, float, int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";
import { Count } from "./Count.js";
import { Triangle } from "./Triangle.js";

export class GeometryClustering {
    public static Clusterize(count: Count, triangles: Triangle[]): Cluster[] {
        const self = GeometryClustering;
        const clusters: Cluster[] = [];
        const unused: Triangle[] = [...triangles];
        let suggestion: Undefinable<Triangle> = undefined;
        while (unused.length > 0) {
            const group: Triangle[] = [];
            const candidates: Triangle[] = [];
            let center: Undefinable<ClusterCenter> = undefined;
            const first: Triangle = self.PopFirst(suggestion, unused);
            suggestion = undefined;
            center = self.Register(first, group, candidates, center);
            while (group.length < ClusterTrianglesLimit && unused.length > 0) {
                if (candidates.length === 0) {
                    const { index, nearest } = self.FindNearest(
                        /*center*/ first.vertices[0].position,
                        unused,
                    );
                    swapRemove(unused, index);
                    center = self.Register(nearest, group, candidates, center);
                    continue;
                }
                const { index, nearest } = self.FindNearest(
                    /*center*/ first.vertices[0].position,
                    candidates,
                );
                swapRemove(candidates, index);
                const nearestInUnusedAt: int = unused.indexOf(nearest);
                if (nearestInUnusedAt === -1) {
                    continue;
                }
                swapRemove(unused, nearestInUnusedAt);
                center = self.Register(nearest, group, candidates, center);
            }

            for (let i: int = 0; i < candidates.length; i++) {
                const candidate: Triangle = candidates[i];
                if (!unused.includes(candidate)) {
                    continue;
                }
                suggestion = candidate;
                break;
            }

            clusters.push(new Cluster(count, group, center));

            suggestion = self.FindNearest(
                clusters[0].bounds.center,
                candidates.filter((candidate: Triangle) =>
                    unused.includes(candidate),
                ),
            ).nearest;
        }
        return clusters;
    }

    private static PopFirst(
        suggestion: Undefinable<Triangle>,
        unused: Triangle[],
    ): Triangle {
        if (!suggestion) {
            let index: Undefinable<int> = undefined;
            for (let i: int = 0; i < unused.length; i++) {
                if (unused[i].isBorder()) {
                    index = i;
                    break;
                }
            }
            if (index === undefined) {
                return unused.pop()!;
            }
            const first: Triangle = unused[index];
            swapRemove(unused, index);
            return first;
        }
        const suggestionInUnusedAt: int = unused.indexOf(suggestion);
        assert(suggestionInUnusedAt !== -1);
        swapRemove(unused, suggestionInUnusedAt);
        return suggestion;
    }

    private static Register(
        triangle: Triangle,
        cluster: Triangle[],
        candidates: Triangle[],
        center?: ClusterCenter,
    ): ClusterCenter {
        cluster.push(triangle);
        candidates.push(...triangle.getAdjacent());
        return GeometryClustering.RecomputeCenter(triangle, center);
    }

    private static RecomputeCenter(
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

    private static FindNearest(
        target: Vec3 | ClusterCenter,
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
                    .sub(target instanceof Vec3 ? target : target.center)
                    .lengthQuadratic(),
                candidate.vertices[1].position
                    .clone()
                    .sub(target instanceof Vec3 ? target : target.center)
                    .lengthQuadratic(),
                candidate.vertices[2].position
                    .clone()
                    .sub(target instanceof Vec3 ? target : target.center)
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

/*
    private clusterize(count: Count, triangles: Triangle[]): Cluster[] {
        const clusters: Cluster[] = [];
        const unused: Triangle[] = [...triangles];
        let suggestions: Undefinable<Triangle[]> = undefined;
        while (unused.length > 0) {
            const group: Triangle[] = [];
            const candidates: Triangle[] = suggestions || [];
            let center: Undefinable<ClusterCenter> = undefined;
            const first: Triangle = this.popFirst(suggestions, unused);
            suggestions = undefined;
            center = this.register(first, group, candidates, center);
            while (group.length < ClusterTrianglesLimit && unused.length > 0) {
                if (candidates.length === 0) {
                    const { index, nearest } = this.findNearest(
                        center.center,
                        unused,
                    );
                    swapRemove(unused, index);
                    center = this.register(nearest, group, candidates, center);
                    continue;
                }
                const { index, nearest } = this.findNearest(
                    center.center,
                    candidates,
                );
                swapRemove(candidates, index);
                const nearestInUnusedAt: int = unused.indexOf(nearest);
                if (nearestInUnusedAt === -1) {
                    continue;
                }
                swapRemove(unused, nearestInUnusedAt);
                center = this.register(nearest, group, candidates, center);
            }*/
/*
            for (let i: int = 0; i < candidates.length; i++) {
                const candidate: Triangle = candidates[i];
                if (!unused.includes(candidate)) {
                    continue;
                }
                suggestion = candidate;
                break;
            }
            */
/*
            //log(candidates.length);
            suggestions = candidates;
            clusters.push(new Cluster(count, group, center));
        }
        return clusters;
    }

    private popFirst(
        suggestions: Undefinable<Triangle[]>,
        unused: Triangle[],
    ): Triangle {
        if (!suggestions) {
            return unused.pop()!;
        }
        while (suggestions.length > 0) {
            const { index, nearest } = this.findNearest(
                center.center,
                candidates,
            );
        }
        for (let i: int = 0; i < suggestions.length; i++) {
            const candidate: Triangle = suggestions[i];
            if (!unused.includes(candidate)) {
                continue;
            }
            const suggestionInUnusedAt: int = unused.indexOf(candidate);
            assert(suggestionInUnusedAt !== -1);
            swapRemove(unused, suggestionInUnusedAt);
            return candidate;
        }
        return unused.pop()!;
    }
    */
