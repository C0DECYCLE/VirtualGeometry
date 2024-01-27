/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterTrianglesLimit } from "../constants.js";
import { ClusterId } from "../core.type.js";
import { assert } from "../utilities/utils.js";
import { Count } from "./Count.js";
import { Triangle } from "./Triangle.js";

export class Cluster {
    public readonly id: ClusterId;
    public readonly triangles: Triangle[];

    public constructor(count: Count, triangles: Triangle[]) {
        assert(triangles.length <= ClusterTrianglesLimit);
        this.id = count.registerCluster();
        this.triangles = triangles;
    }
}
