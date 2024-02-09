/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { GeometryId, GeometryKey } from "../core.type.js";
import { Virtual } from "../generator/Virtual.js";

export class Geometry {
    public readonly id: GeometryId;
    public readonly key: GeometryKey;

    public constructor(id: GeometryId, key: GeometryKey) {
        this.id = id;
        this.key = key;
    }

    public static FromVirtual(virtual: Virtual): Geometry {
        return new Geometry(virtual.id, virtual.key);
    }
}
