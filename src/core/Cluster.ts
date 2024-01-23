/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterTrianglesLimit } from "../constants.js";
import { ClusterBounds, ClusterCenter, ClusterId } from "../core.type.js";
import { assert } from "../utilities/utils.js";
import { float, int } from "../utils.type.js";
import { ClusterCount, GeometryCount, GeometryHandlerCount } from "./Counts.js";
import { Triangle } from "./Triangle.js";

export class Cluster {
    public readonly inHandlerId: ClusterId;
    public readonly inGeometryId: ClusterId;
    public readonly count: ClusterCount;

    public readonly triangles: Triangle[];
    public readonly bounds: ClusterBounds;

    public constructor(
        handlerCount: GeometryHandlerCount,
        geometryCount: GeometryCount,
        triangles: Triangle[],
        center: ClusterCenter,
    ) {
        this.inHandlerId = handlerCount.registerCluster();
        this.inGeometryId = geometryCount.registerCluster();
        this.count = new ClusterCount(0, 0);
        assert(triangles.length <= ClusterTrianglesLimit);
        this.triangles = this.registerTriangles(triangles);
        this.bounds = this.computeBounds(center);
    }

    private registerTriangles(triangles: Triangle[]): Triangle[] {
        for (let i: int = 0; i < triangles.length; i++) {
            triangles[i].register(this.count);
            triangles[i].vertices[0].register(this.count);
            triangles[i].vertices[1].register(this.count);
            triangles[i].vertices[2].register(this.count);
        }
        return triangles;
    }

    private computeBounds(center: ClusterCenter): ClusterBounds {
        let radiusQuadratic: float = 0;
        for (let i: int = 0; i < this.triangles.length; i++) {
            const triangle: Triangle = this.triangles[i];
            radiusQuadratic = Math.max(
                radiusQuadratic,
                triangle.vertices[0].position
                    .clone()
                    .sub(center.center)
                    .lengthQuadratic(),
                triangle.vertices[1].position
                    .clone()
                    .sub(center.center)
                    .lengthQuadratic(),
                triangle.vertices[2].position
                    .clone()
                    .sub(center.center)
                    .lengthQuadratic(),
            );
        }
        return {
            center: center.center,
            radius: Math.sqrt(radiusQuadratic),
        } as ClusterBounds;
    }
}
