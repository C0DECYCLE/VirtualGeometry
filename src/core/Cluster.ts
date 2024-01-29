/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterTrianglesLimit } from "../constants.js";
import { ClusterBounds, ClusterCenter } from "../core.type.js";
import { assert } from "../utilities/utils.js";
import { float, int } from "../utils.type.js";
import { Count } from "./Count.js";
import { Triangle } from "./Triangle.js";

export class Cluster {
    public readonly triangles: Triangle[];
    public readonly bounds: ClusterBounds;

    public constructor(
        count: Count,
        triangles: Triangle[],
        center: ClusterCenter,
    ) {
        assert(triangles.length <= ClusterTrianglesLimit);
        count.registerCluster();
        this.triangles = triangles;
        this.bounds = this.computeBounds(center);
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
