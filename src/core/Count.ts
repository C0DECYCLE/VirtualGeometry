/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { ClusterId, GeometryId, TriangleId, VertexId } from "../core.type";
import { int } from "../utils.type";

export class Count {
    private readonly track: {
        geometries: int;
        clusters: int;
        triangles: int;
        vertices: int;
    };

    public constructor(
        geometries: int,
        clusters: int,
        triangles: int,
        vertices: int,
    ) {
        this.track = {
            geometries: geometries,
            clusters: clusters,
            triangles: triangles,
            vertices: vertices,
        };
    }

    public get geometries(): int {
        return this.track.geometries;
    }

    public get clusters(): int {
        return this.track.clusters;
    }

    public get triangles(): int {
        return this.track.triangles;
    }

    public get vertices(): int {
        return this.track.vertices;
    }

    public registerGeometry(): GeometryId {
        this.track.geometries++;
        return this.track.geometries - 1;
    }

    public registerCluster(): ClusterId {
        this.track.clusters++;
        return this.track.clusters - 1;
    }

    public registerTriangle(): TriangleId {
        this.track.triangles++;
        return this.track.triangles - 1;
    }

    public registerVertex(): VertexId {
        this.track.vertices++;
        return this.track.vertices - 1;
    }
}
