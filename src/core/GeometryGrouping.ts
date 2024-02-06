/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { ClusterGroupingLimit } from "../constants.js";
import { VertexId } from "../core.type.js";
import { assert, swapRemove } from "../utilities/utils.js";
import { Undefinable, int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";

export class GeometryGrouping {
    public static Group(clusters: Cluster[]): Cluster[][] {
        const self = GeometryGrouping;
        const groups: Cluster[][] = [];
        const unused: Cluster[] = [...clusters];
        let suggestion: Undefinable<Cluster> = undefined;
        while (unused.length > 0) {
            const group: Cluster[] = [];
            const first: Cluster = self.PopFirst(suggestion, unused);
            suggestion = undefined;
            group.push(first);
            while (group.length < ClusterGroupingLimit && unused.length > 0) {
                const { index, highest } = self.FindHighest(group, unused);
                swapRemove(unused, index);
                group.push(highest);
            }
            groups.push(group);
            if (unused.length > 0) {
                suggestion = self.FindHighest(group, unused).highest;
            }
        }
        return groups;
    }

    private static PopFirst(
        suggestion: Undefinable<Cluster>,
        unused: Cluster[],
    ): Cluster {
        //either suggestion
        if (suggestion) {
            const suggestionInUnusedAt: int = unused.indexOf(suggestion);
            assert(suggestionInUnusedAt !== -1);
            swapRemove(unused, suggestionInUnusedAt);
            return suggestion;
        }
        //or last
        return unused.pop()!;
    }

    private static FindHighest(
        group: Cluster[],
        candidates: Cluster[],
    ): {
        index: int;
        highest: Cluster;
    } {
        assert(group.length > 0 && candidates.length > 0);
        let index: Undefinable<int> = undefined;
        let highest: Undefinable<Cluster> = undefined;
        let score: int = -1;
        const border: Set<VertexId> = new Set<VertexId>(
            group.flatMap((cluster: Cluster) => [...cluster.border]),
        );
        for (let i: int = 0; i < candidates.length; i++) {
            const cluster: Cluster = candidates[i];
            let matching: int = 0;
            cluster.border.forEach((id: VertexId) => {
                if (border.has(id)) {
                    matching++;
                }
            });
            if (matching > score) {
                index = i;
                highest = cluster;
                score = matching;
            }
        }
        assert(index !== undefined && highest);
        return { index, highest };
    }
}
