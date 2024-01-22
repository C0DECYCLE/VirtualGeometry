/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterLimit } from "../constants.js";
import { ClusterId } from "../core.type.js";
import { assert } from "../utilities/utils.js";
import { Triangle } from "./Triangle.js";

export class Cluster {
    public readonly globalId: ClusterId;
    public readonly localId: ClusterId;

    public readonly triangles: Triangle[];

    public constructor(
        globalId: ClusterId,
        localId: ClusterId,
        triangles: Triangle[],
    ) {
        this.globalId = globalId;
        this.localId = localId;
        assert(triangles.length <= ClusterLimit);
        this.triangles = triangles;
    }
}
