/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { ClusterGroupingLimit } from "../constants.js";
import { ClusterGroup, EdgeIdentifier, VertexId } from "../core.type.js";
import { assert, swapRemove } from "../utilities/utils.js";
import { Undefinable, int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";
import { Count } from "./Count.js";
import { Edge } from "./Edge.js";
import { Triangle } from "./Triangle.js";

export class Grouping {
    public static Group(clusters: Cluster[]): ClusterGroup[] {
        const groups: ClusterGroup[] = [];
        const unused: Cluster[] = [...clusters];
        let suggestion: Undefinable<Cluster> = undefined;
        while (unused.length > 0) {
            const group: ClusterGroup = [];
            const first: Cluster = Grouping.PopFirst(suggestion, unused);
            suggestion = undefined;
            group.push(first);
            while (group.length < ClusterGroupingLimit && unused.length > 0) {
                // prettier-ignore
                const { index, highest } = Grouping.FindHighest(group, unused);
                swapRemove(unused, index);
                group.push(highest);
            }
            groups.push(group);
            if (unused.length > 0) {
                suggestion = Grouping.FindHighest(group, unused).highest;
            }
        }
        return groups;
    }

    private static PopFirst(
        suggestion: Undefinable<Cluster>,
        unused: Cluster[],
    ): Cluster {
        if (suggestion) {
            const suggestionInUnusedAt: int = unused.indexOf(suggestion);
            assert(suggestionInUnusedAt !== -1);
            swapRemove(unused, suggestionInUnusedAt);
            return suggestion;
        }
        return unused.pop()!;
    }

    private static FindHighest(
        group: ClusterGroup,
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
            group.flatMap((cluster: Cluster) => [...cluster.border!]),
        );
        for (let i: int = 0; i < candidates.length; i++) {
            const cluster: Cluster = candidates[i];
            let matching: int = 0;
            cluster.border!.forEach((id: VertexId) => {
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

    public static Merge(
        count: Count,
        group: ClusterGroup,
    ): { triangles: Triangle[]; edges: Map<EdgeIdentifier, Edge> } {
        assert(group.length > 0);
        const triangles: Triangle[] = [];
        const edges: Map<EdgeIdentifier, Edge> = new Map<
            EdgeIdentifier,
            Edge
        >();
        for (let i: int = 0; i < group.length; i++) {
            const cluster: Cluster = group[i];
            for (let j: int = 0; j < cluster.triangles.length; j++) {
                const triangle: Triangle = new Triangle(count, [
                    ...cluster.triangles[j].vertices,
                ]);
                triangles.push(triangle);
                triangle.registerEdgesToMap(edges);
            }
        }
        return { triangles, edges };
    }
}
