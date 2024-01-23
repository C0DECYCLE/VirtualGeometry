/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int } from "../utils.type";

export class ClusterCount {
    protected readonly track: {
        triangles: int;
        vertices: int;
    };

    public constructor(triangles: int, vertices: int) {
        this.track = {
            triangles: triangles,
            vertices: vertices,
        };
    }

    public get triangles(): int {
        return this.track.triangles;
    }

    public get vertices(): int {
        return this.track.vertices;
    }

    public incTriangles(): int {
        this.track.triangles++;
        return this.track.triangles - 1;
    }

    public incVertices(): int {
        this.track.vertices++;
        return this.track.vertices - 1;
    }
}

export class GeometryCount extends ClusterCount {
    protected override readonly track: {
        clusters: int;
        triangles: int;
        vertices: int;
    };

    public constructor(clusters: int, triangles: int, vertices: int) {
        super(triangles, vertices);
        this.track = {
            clusters: clusters,
            triangles: triangles,
            vertices: vertices,
        };
    }

    public get clusters(): int {
        return this.track.clusters;
    }

    public incClusters(): int {
        this.track.clusters++;
        return this.track.clusters - 1;
    }
}

export class GeometryHandlerCount extends GeometryCount {
    protected override readonly track: {
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
        super(clusters, triangles, vertices);
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

    public incGeometries(): int {
        this.track.geometries++;
        return this.track.geometries - 1;
    }
}
