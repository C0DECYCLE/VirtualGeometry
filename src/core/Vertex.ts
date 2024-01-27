/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { VertexId } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert } from "../utilities/utils.js";
import { float } from "../utils.type.js";
import { Count } from "./Count.js";

export class Vertex {
    public readonly id: VertexId;
    public readonly position: Vec3;

    public constructor(count: Count, data: float[] | Float32Array) {
        assert(data.length >= 3);
        this.id = count.registerVertex();
        this.position = new Vec3(data[0], data[1], data[2]);
    }
}
