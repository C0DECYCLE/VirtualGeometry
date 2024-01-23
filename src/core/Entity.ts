/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { GeometryKey } from "../core.type.js";
import { Vec3 } from "../utilities/Vec3.js";

export class Entity {
    public readonly position: Vec3;
    public readonly key: GeometryKey;

    public constructor(position: Vec3, key: GeometryKey) {
        this.position = position;
        this.key = key;
    }
}
