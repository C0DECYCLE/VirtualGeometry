/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterGroupingLimit } from "../constants.js";
import { assert, swapRemove } from "../utilities/utils.js";
import { Nullable, Undefinable, float, int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";

export class GeometryGrouping {
    public static Group(clusters: Cluster[]): Cluster[][] {
        const self = GeometryGrouping;
        const groups: Cluster[][] = [];
        const unused: Cluster[] = [...clusters];

        let suggestion: Undefinable<Cluster> = undefined;
        while (unused.length > 0) {
            const group: Cluster[] = [];

            const first: Cluster = suggestion || unused.shift()!;
            if (suggestion) {
                const suggestionInUnusedAt: int = unused.indexOf(suggestion);
                assert(suggestionInUnusedAt !== -1);
                swapRemove(unused, suggestionInUnusedAt);
                suggestion = undefined;
            }
            group.push(first);

            while (group.length < ClusterGroupingLimit && unused.length > 0) {
                const { index, nearest } = self.FindNearestCluster(
                    unused,
                    first,
                );
                swapRemove(unused, index);
                group.push(nearest);

                //group.push(unused.shift()!);
            }

            if (unused.length > 0) {
                suggestion = self.FindNearestCluster(unused, first).nearest;
            }

            groups.push(group);
        }
        return groups;
    }

    private static FindNearestCluster(
        clusters: Cluster[],
        target: Cluster,
    ): { index: int; nearest: Cluster } {
        assert(clusters.length > 0);
        let nearest: Nullable<Cluster> = null;
        let distance: float = Infinity;
        let index: Undefinable<int> = undefined;
        for (let i: int = 0; i < clusters.length; i++) {
            const cluster: Cluster = clusters[i];
            const dist: float = cluster.bounds.center
                .clone()
                .sub(target.bounds.center)
                .length();
            if (dist < distance) {
                nearest = cluster;
                distance = dist;
                index = i;
            }
        }
        assert(nearest && index !== undefined);
        return { index, nearest };
    }
}
