/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ClusterId, EntityIndex, GeometryKey } from "../core.type.js";
import { EntityHandler } from "../handlers/EntityHandler.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert } from "../utilities/utils.js";
import { Nullable, Undefinable, float } from "../utils.type.js";

export class Entity {
    private readonly position: Vec3;
    public readonly key: GeometryKey;

    private handler: Nullable<EntityHandler>;
    private index: Undefinable<EntityIndex>;

    public constructor(key: GeometryKey) {
        this.position = new Vec3();
        this.key = key;
        this.handler = null;
        this.index = undefined;
    }

    public getPosition(): Vec3 {
        return this.position.clone();
    }

    public setPosition(x: Vec3 | float, y?: float, z?: float): void {
        this.position.set(x, y, z);
        if (!this.handler) {
            return;
        }
        assert(this.index !== undefined);
        this.handler.registerChange(this.index, "UPDATE");
    }

    public add(handler: EntityHandler): void {
        assert(!this.handler);
        this.handler = handler;
        this.index = this.handler.append(this);
    }

    public getRootId(): ClusterId {
        assert(this.handler);
        return this.handler.getRootId(this.key);
    }
}
