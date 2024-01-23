/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { VertexId } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert } from "../utilities/utils.js";
import { Undefinable } from "../utils.type.js";
import { ClusterCount, GeometryCount, GeometryHandlerCount } from "./Counts.js";

export class Vertex {
    public readonly inHandlerId: VertexId;
    public readonly inGeometryId: VertexId;
    public inClusterId: Undefinable<VertexId>;

    public readonly position: Vec3;

    public constructor(
        handlerCount: GeometryHandlerCount,
        geometryCount: GeometryCount,
        clusterCount: Undefinable<ClusterCount>,
        data: Float32Array,
    ) {
        this.inHandlerId = handlerCount.registerVertex();
        this.inGeometryId = geometryCount.registerVertex();
        if (clusterCount) {
            this.inClusterId = clusterCount.registerVertex();
        }
        assert(data.length >= 3);
        this.position = new Vec3(data[0], data[1], data[2]);
    }

    public register(clusterCount: ClusterCount): void {
        if (this.inClusterId !== undefined) {
            return;
        }
        this.inClusterId = clusterCount.registerVertex();
    }
}
