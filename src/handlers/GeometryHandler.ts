/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterId, GeometryKey } from "../core.type.js";
import { Nullable } from "../utils.type.js";
import { Geometry } from "../components/Geometry.js";
import { Generator } from "../generator/Generator.js";
import { assert } from "../utilities/utils.js";

export class GeometryHandler {
    public readonly geometries: Map<GeometryKey, Geometry>;

    public clustersBuffer: Nullable<GPUBuffer>;
    public trianglesBuffer: Nullable<GPUBuffer>;
    public verticesBuffer: Nullable<GPUBuffer>;

    public rootIds: Nullable<Map<GeometryKey, ClusterId>>;

    public constructor() {
        this.geometries = new Map<GeometryKey, Geometry>();
        this.clustersBuffer = null;
        this.trianglesBuffer = null;
        this.verticesBuffer = null;
        this.rootIds = null;
    }

    public async import(key: GeometryKey, path: string): Promise<void> {
        this.geometries.set(key, await Generator.Generate(key, path));
    }

    public prepare(device: GPUDevice): void {
        const { clusters, triangles, vertices, rootIds } =
            Generator.Compact(device);
        this.clustersBuffer = clusters;
        this.trianglesBuffer = triangles;
        this.verticesBuffer = vertices;
        this.rootIds = rootIds;
    }

    public exists(key: GeometryKey): boolean {
        return this.geometries.has(key);
    }

    public getRootId(key: GeometryKey): ClusterId {
        assert(this.rootIds);
        return this.rootIds.get(key)!;
    }
}
