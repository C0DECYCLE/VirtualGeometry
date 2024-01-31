/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterGroupingLimit } from "../constants.js";
import { Cluster } from "./Cluster.js";

export class GeometryGrouping {
    public static Group(clusters: Cluster[]): Cluster[][] {
        const groups: Cluster[][] = [];
        const unused: Cluster[] = [...clusters];
        while (unused.length > 0) {
            const group: Cluster[] = [];
            while (group.length < ClusterGroupingLimit && unused.length > 0) {
                group.push(unused.shift()!);
            }
            groups.push(group);
        }
        return groups;
    }
}
