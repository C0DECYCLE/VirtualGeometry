/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { Bounding, GeometryId, GeometryKey } from "../core.type.js";
import { Virtual } from "../generator/Virtual.js";
import { assert } from "../utilities/utils.js";
import { int } from "../utils.type.js";

export class Geometry {
    public readonly id: GeometryId;
    public readonly key: GeometryKey;
    public readonly bounding: Bounding;

    public readonly leaveTrianglesCount: int;

    public constructor(
        id: GeometryId,
        key: GeometryKey,
        bounding: Bounding,
        leaveTrianglesCount: int,
    ) {
        this.id = id;
        this.key = key;
        this.bounding = bounding;
        this.leaveTrianglesCount = leaveTrianglesCount;
    }

    public static FromVirtual(virtual: Virtual): Geometry {
        assert(virtual.bounding && virtual.leaveTrianglesCount);
        return new Geometry(
            virtual.id,
            virtual.key,
            virtual.bounding,
            virtual.leaveTrianglesCount,
        );
    }
}
