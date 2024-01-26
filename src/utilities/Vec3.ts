/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float, FloatArray } from "../../types/utils.type.js";
import { Mat4 } from "./Mat4.js";

export class Vec3 {
    public static Cache: Vec3 = new Vec3();

    private _x: float;
    private _y: float;
    private _z: float;

    public isDirty: boolean;

    public get x(): float {
        return this._x;
    }

    public set x(value: float) {
        this._x = value;
        this.isDirty = true;
    }

    public get y(): float {
        return this._y;
    }

    public set y(value: float) {
        this._y = value;
        this.isDirty = true;
    }

    public get z(): float {
        return this._z;
    }

    public set z(value: float) {
        this._z = value;
        this.isDirty = true;
    }

    public constructor(x: Vec3 | float = 0, y: float = 0, z: float = 0) {
        this.isDirty = false;
        this.set(x, y, z);
    }

    public set(x: Vec3 | float, y?: float, z?: float): Vec3 {
        if (x instanceof Vec3) {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === undefined || z === undefined) {
            z = x;
            y = x;
        }
        this.x = x;
        this.y = y;
        this.z = z;
        return this;
    }

    public add(x: Vec3 | float, y?: float, z?: float): Vec3 {
        if (x instanceof Vec3) {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === undefined || z === undefined) {
            z = x;
            y = x;
        }
        if (x === 0 && y === 0 && z === 0) {
            return this;
        }
        this.x += x;
        this.y += y;
        this.z += z;
        return this;
    }

    public sub(x: Vec3 | float, y?: float, z?: float): Vec3 {
        if (x instanceof Vec3) {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === undefined || z === undefined) {
            z = x;
            y = x;
        }
        if (x === 0 && y === 0 && z === 0) {
            return this;
        }
        this.x -= x;
        this.y -= y;
        this.z -= z;
        return this;
    }

    public scale(x: Vec3 | float, y?: float, z?: float): Vec3 {
        if (x instanceof Vec3) {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === undefined || z === undefined) {
            z = x;
            y = x;
        }
        this.x *= x;
        this.y *= y;
        this.z *= z;
        return this;
    }

    public divide(x: Vec3 | float, y?: float, z?: float): Vec3 {
        if (x instanceof Vec3) {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === undefined || z === undefined) {
            z = x;
            y = x;
        }
        this.x /= x;
        this.y /= y;
        this.z /= z;
        return this;
    }

    public lengthQuadratic(): float {
        return this.x * this.x + this.y * this.y + this.z * this.z;
    }

    public length(): float {
        return Math.sqrt(this.lengthQuadratic());
    }

    public normalize(): Vec3 {
        return this.divide(this.length());
    }

    public dot(x: Vec3 | float, y?: float, z?: float): float {
        if (x instanceof Vec3) {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === undefined || z === undefined) {
            z = x;
            y = x;
        }
        return this.x * x + this.y * y + this.z * z;
    }

    public cross(b: Vec3, a: Vec3 = this): Vec3 {
        return this.copy(
            Vec3.Cache.set(
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x,
            ),
        );
    }

    public applyMat(mat: Mat4): Vec3 {
        const w =
            1 /
            (mat.values[3] * this.x +
                mat.values[7] * this.y +
                mat.values[11] * this.z +
                mat.values[15]);

        this.x =
            (mat.values[0] * this.x +
                mat.values[4] * this.y +
                mat.values[8] * this.z +
                mat.values[12]) *
            w;
        this.y =
            (mat.values[1] * this.x +
                mat.values[5] * this.y +
                mat.values[9] * this.z +
                mat.values[13]) *
            w;
        this.z =
            (mat.values[2] * this.x +
                mat.values[6] * this.y +
                mat.values[10] * this.z +
                mat.values[14]) *
            w;
        return this;
    }

    public copy(v: Vec3): Vec3 {
        return this.set(v.x, v.y, v.z);
    }

    public store(target: FloatArray, offset: int = 0): Vec3 {
        target[offset + 0] = this.x;
        target[offset + 1] = this.y;
        target[offset + 2] = this.z;
        return this;
    }

    public clone(): Vec3 {
        return new Vec3(this.x, this.y, this.z);
    }

    public toJSON(): Object {
        return {
            x: this.x,
            y: this.y,
            z: this.z,
        };
    }

    public toArray(): [float, float, float] {
        return [this.x, this.y, this.z];
    }

    public toString(): string {
        return JSON.stringify(this);
    }

    public static Dot(a: Vec3, b: Vec3): float {
        return a.dot(b);
    }

    public static Cross(a: Vec3, b: Vec3): Vec3 {
        return a.clone().cross(b);
    }
}
