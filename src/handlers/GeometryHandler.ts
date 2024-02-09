/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { GeometryKey } from "../core.type.js";
import { Nullable, int } from "../utils.type.js";
import { Geometry } from "../components/Geometry.js";
import { Generator } from "../generator/Generator.js";

export class GeometryHandler {
    public readonly geometries: Geometry[];

    public clustersBuffer: Nullable<GPUBuffer>;
    public trianglesBuffer: Nullable<GPUBuffer>;
    public verticesBuffer: Nullable<GPUBuffer>;

    public constructor() {
        this.geometries = [];
        this.clustersBuffer = null;
        this.trianglesBuffer = null;
        this.verticesBuffer = null;
    }

    public async import(key: GeometryKey, path: string): Promise<void> {
        this.geometries.push(await Generator.Generate(key, path));
    }

    public prepare(device: GPUDevice): void {
        const { clusters, triangles, vertices } = Generator.Compact(device);
        this.clustersBuffer = clusters;
        this.trianglesBuffer = triangles;
        this.verticesBuffer = vertices;
    }

    public exists(key: GeometryKey): boolean {
        for (let i: int = 0; i < this.geometries.length; i++) {
            if (this.geometries[i].key === key) {
                return true;
            }
        }
        return false;
    }
}
