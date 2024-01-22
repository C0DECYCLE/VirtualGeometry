/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { VertexId } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert } from "../utilities/utils.js";

export class Vertex {
    public readonly globalId: VertexId;
    public readonly localId: VertexId;

    public readonly position: Vec3;

    public constructor(
        globalId: VertexId,
        localId: VertexId,
        data: Float32Array,
    ) {
        this.globalId = globalId;
        this.localId = localId;
        assert(data.length >= 3);
        this.position = new Vec3(data[0], data[1], data[2]);
    }
}
