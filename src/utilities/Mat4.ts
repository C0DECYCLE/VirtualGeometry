/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float, FloatArray } from "../../types/utilities/utils.type.js";
import { Vec3 } from "./Vec3.js";

export class Mat4 {
    public static Cache: Mat4 = new Mat4();

    public readonly values: Float32Array | Float64Array;
    public readonly isFloat64: boolean;

    public constructor(isFloat64: boolean = false) {
        this.values = isFloat64 ? new Float64Array(16) : new Float32Array(16);
        this.isFloat64 = isFloat64;
        this.reset();
    }

    public set(...floats: float[]): Mat4 {
        this.values[0] = floats[0];
        this.values[1] = floats[1];
        this.values[2] = floats[2];
        this.values[3] = floats[3];
        this.values[4] = floats[4];
        this.values[5] = floats[5];
        this.values[6] = floats[6];
        this.values[7] = floats[7];
        this.values[8] = floats[8];
        this.values[9] = floats[9];
        this.values[10] = floats[10];
        this.values[11] = floats[11];
        this.values[12] = floats[12];
        this.values[13] = floats[13];
        this.values[14] = floats[14];
        this.values[15] = floats[15];
        return this;
    }

    public reset(): Mat4 {
        // prettier-ignore
        return this.set(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        );
    }

    public translate(x: Vec3 | float, y?: float, z?: float): Mat4 {
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
        const n00: float = this.values[0];
        const n01: float = this.values[1];
        const n02: float = this.values[2];
        const n03: float = this.values[3];
        const n10: float = this.values[4];
        const n11: float = this.values[5];
        const n12: float = this.values[6];
        const n13: float = this.values[7];
        const n20: float = this.values[8];
        const n21: float = this.values[9];
        const n22: float = this.values[10];
        const n23: float = this.values[11];
        const n30: float = this.values[12];
        const n31: float = this.values[13];
        const n32: float = this.values[14];
        const n33: float = this.values[15];

        // prettier-ignore
        this.values[12] = n00 * (x as float) + n10 * (y as float) + n20 * (z as float) + n30;
        // prettier-ignore
        this.values[13] = n01 * (x as float) + n11 * (y as float) + n21 * (z as float) + n31;
        // prettier-ignore
        this.values[14] = n02 * (x as float) + n12 * (y as float) + n22 * (z as float) + n32;
        // prettier-ignore
        this.values[15] = n03 * (x as float) + n13 * (y as float) + n23 * (z as float) + n33;

        return this;
    }

    public rotateX(radian: float): Mat4 {
        if (radian === 0) {
            return this;
        }
        const n10: float = this.values[4];
        const n11: float = this.values[5];
        const n12: float = this.values[6];
        const n13: float = this.values[7];
        const n20: float = this.values[8];
        const n21: float = this.values[9];
        const n22: float = this.values[10];
        const n23: float = this.values[11];

        const c: float = Math.cos(radian);
        const s: float = Math.sin(radian);

        this.values[4] = c * n10 + s * n20;
        this.values[5] = c * n11 + s * n21;
        this.values[6] = c * n12 + s * n22;
        this.values[7] = c * n13 + s * n23;
        this.values[8] = c * n20 - s * n10;
        this.values[9] = c * n21 - s * n11;
        this.values[10] = c * n22 - s * n12;
        this.values[11] = c * n23 - s * n13;

        return this;
    }

    public rotateY(radian: float): Mat4 {
        if (radian === 0) {
            return this;
        }
        const n00: float = this.values[0];
        const n01: float = this.values[1];
        const n02: float = this.values[2];
        const n03: float = this.values[3];
        const n20: float = this.values[8];
        const n21: float = this.values[9];
        const n22: float = this.values[10];
        const n23: float = this.values[11];

        const c: float = Math.cos(radian);
        const s: float = Math.sin(radian);

        this.values[0] = c * n00 - s * n20;
        this.values[1] = c * n01 - s * n21;
        this.values[2] = c * n02 - s * n22;
        this.values[3] = c * n03 - s * n23;
        this.values[8] = c * n20 + s * n00;
        this.values[9] = c * n21 + s * n01;
        this.values[10] = c * n22 + s * n02;
        this.values[11] = c * n23 + s * n03;

        return this;
    }

    public rotateZ(radian: float): Mat4 {
        if (radian === 0) {
            return this;
        }
        const n00: float = this.values[0];
        const n01: float = this.values[1];
        const n02: float = this.values[2];
        const n03: float = this.values[3];
        const n10: float = this.values[4];
        const n11: float = this.values[5];
        const n12: float = this.values[6];
        const n13: float = this.values[7];

        const c: float = Math.cos(radian);
        const s: float = Math.sin(radian);

        this.values[0] = c * n00 + s * n10;
        this.values[1] = c * n01 + s * n11;
        this.values[2] = c * n02 + s * n12;
        this.values[3] = c * n03 + s * n13;
        this.values[4] = c * n10 - s * n00;
        this.values[5] = c * n11 - s * n01;
        this.values[6] = c * n12 - s * n02;
        this.values[7] = c * n13 - s * n03;

        return this;
    }

    public rotateAxis(normal: Vec3, radian: float): Mat4 {
        if (radian === 0) {
            return this;
        }
        const x: float = normal.x;
        const y: float = normal.y;
        const z: float = normal.z;

        const xx: float = x * x;
        const yy: float = y * y;
        const zz: float = z * z;

        const c: float = Math.cos(radian);
        const s: float = Math.sin(radian);
        const cI: float = 1.0 - c;

        const r00: float = xx + (1 - xx) * c;
        const r01: float = x * y * cI + z * s;
        const r02: float = x * z * cI - y * s;
        const r10: float = x * y * cI - z * s;
        const r11: float = yy + (1 - yy) * c;
        const r12: float = y * z * cI + x * s;
        const r20: float = x * z * cI + y * s;
        const r21: float = y * z * cI - x * s;
        const r22: float = zz + (1 - zz) * c;

        const n00: float = this.values[0];
        const n01: float = this.values[1];
        const n02: float = this.values[2];
        const n03: float = this.values[3];
        const n10: float = this.values[4];
        const n11: float = this.values[5];
        const n12: float = this.values[6];
        const n13: float = this.values[7];
        const n20: float = this.values[8];
        const n21: float = this.values[9];
        const n22: float = this.values[10];
        const n23: float = this.values[11];

        this.values[0] = r00 * n00 + r01 * n10 + r02 * n20;
        this.values[1] = r00 * n01 + r01 * n11 + r02 * n21;
        this.values[2] = r00 * n02 + r01 * n12 + r02 * n22;
        this.values[3] = r00 * n03 + r01 * n13 + r02 * n23;
        this.values[4] = r10 * n00 + r11 * n10 + r12 * n20;
        this.values[5] = r10 * n01 + r11 * n11 + r12 * n21;
        this.values[6] = r10 * n02 + r11 * n12 + r12 * n22;
        this.values[7] = r10 * n03 + r11 * n13 + r12 * n23;
        this.values[8] = r20 * n00 + r21 * n10 + r22 * n20;
        this.values[9] = r20 * n01 + r21 * n11 + r22 * n21;
        this.values[10] = r20 * n02 + r21 * n12 + r22 * n22;
        this.values[11] = r20 * n03 + r21 * n13 + r22 * n23;

        return this;
    }

    public scale(x: Vec3 | float, y?: float, z?: float): Mat4 {
        if (x instanceof Vec3) {
            z = x.z;
            y = x.y;
            x = x.x;
        } else if (y === undefined || z === undefined) {
            y = x;
            z = x;
        }
        if (x === 0 && y === 0 && z === 0) {
            return this;
        }
        this.values[0] *= x as float;
        this.values[1] *= x as float;
        this.values[2] *= x as float;
        this.values[3] *= x as float;
        this.values[4] *= y as float;
        this.values[5] *= y as float;
        this.values[6] *= y as float;
        this.values[7] *= y as float;
        this.values[8] *= z as float;
        this.values[9] *= z as float;
        this.values[10] *= z as float;
        this.values[11] *= z as float;
        return this;
    }

    public multiply(b: Mat4, a: Mat4 = this): Mat4 {
        const a00 = a.values[0];
        const a01 = a.values[1];
        const a02 = a.values[2];
        const a03 = a.values[3];
        const a10 = a.values[4];
        const a11 = a.values[5];
        const a12 = a.values[6];
        const a13 = a.values[7];
        const a20 = a.values[8];
        const a21 = a.values[9];
        const a22 = a.values[10];
        const a23 = a.values[11];
        const a30 = a.values[12];
        const a31 = a.values[13];
        const a32 = a.values[14];
        const a33 = a.values[15];

        const b00 = b.values[0];
        const b01 = b.values[1];
        const b02 = b.values[2];
        const b03 = b.values[3];
        const b10 = b.values[4];
        const b11 = b.values[5];
        const b12 = b.values[6];
        const b13 = b.values[7];
        const b20 = b.values[8];
        const b21 = b.values[9];
        const b22 = b.values[10];
        const b23 = b.values[11];
        const b30 = b.values[12];
        const b31 = b.values[13];
        const b32 = b.values[14];
        const b33 = b.values[15];

        this.values[0] = b00 * a00 + b01 * a10 + b02 * a20 + b03 * a30;
        this.values[1] = b00 * a01 + b01 * a11 + b02 * a21 + b03 * a31;
        this.values[2] = b00 * a02 + b01 * a12 + b02 * a22 + b03 * a32;
        this.values[3] = b00 * a03 + b01 * a13 + b02 * a23 + b03 * a33;
        this.values[4] = b10 * a00 + b11 * a10 + b12 * a20 + b13 * a30;
        this.values[5] = b10 * a01 + b11 * a11 + b12 * a21 + b13 * a31;
        this.values[6] = b10 * a02 + b11 * a12 + b12 * a22 + b13 * a32;
        this.values[7] = b10 * a03 + b11 * a13 + b12 * a23 + b13 * a33;
        this.values[8] = b20 * a00 + b21 * a10 + b22 * a20 + b23 * a30;
        this.values[9] = b20 * a01 + b21 * a11 + b22 * a21 + b23 * a31;
        this.values[10] = b20 * a02 + b21 * a12 + b22 * a22 + b23 * a32;
        this.values[11] = b20 * a03 + b21 * a13 + b22 * a23 + b23 * a33;
        this.values[12] = b30 * a00 + b31 * a10 + b32 * a20 + b33 * a30;
        this.values[13] = b30 * a01 + b31 * a11 + b32 * a21 + b33 * a31;
        this.values[14] = b30 * a02 + b31 * a12 + b32 * a22 + b33 * a32;
        this.values[15] = b30 * a03 + b31 * a13 + b32 * a23 + b33 * a33;

        return this;
    }

    public invert(): Mat4 {
        const n11: float = this.values[0];
        const n21: float = this.values[1];
        const n31: float = this.values[2];
        const n41: float = this.values[3];
        const n12: float = this.values[4];
        const n22: float = this.values[5];
        const n32: float = this.values[6];
        const n42: float = this.values[7];
        const n13: float = this.values[8];
        const n23: float = this.values[9];
        const n33: float = this.values[10];
        const n43: float = this.values[11];
        const n14: float = this.values[12];
        const n24: float = this.values[13];
        const n34: float = this.values[14];
        const n44: float = this.values[15];

        // prettier-ignore
        const t11: float = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
        // prettier-ignore
        const t12: float = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
        // prettier-ignore
        const t13: float = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
        // prettier-ignore
        const t14: float = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

        const det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;

        if (det === 0) {
            // prettier-ignore
            return this.set(
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0
            );
        }

        const detInv: float = 1 / det;

        this.values[0] = t11 * detInv;
        // prettier-ignore
        this.values[1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * detInv;
        // prettier-ignore
        this.values[2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * detInv;
        // prettier-ignore
        this.values[3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * detInv;
        this.values[4] = t12 * detInv;
        // prettier-ignore
        this.values[5] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * detInv;
        // prettier-ignore
        this.values[6] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * detInv;
        // prettier-ignore
        this.values[7] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * detInv;
        this.values[8] = t13 * detInv;
        // prettier-ignore
        this.values[9] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * detInv;
        // prettier-ignore
        this.values[10] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * detInv;
        // prettier-ignore
        this.values[11] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * detInv;
        this.values[12] = t14 * detInv;
        // prettier-ignore
        this.values[13] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * detInv;
        // prettier-ignore
        this.values[14] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * detInv;
        // prettier-ignore
        this.values[15] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * detInv;

        return this;
    }

    public aim(pos: Vec3, dir: Vec3, up: Vec3): Mat4 {
        const xAxis: Vec3 = Vec3.Cross(up, dir).normalize();
        const yAxis: Vec3 = Vec3.Cross(dir, xAxis);
        // prettier-ignore
        return this.set(
            xAxis.x, xAxis.y, xAxis.z, 0,
            yAxis.x, yAxis.y, yAxis.z, 0,
            dir.x, dir.y, dir.z, 0,
            pos.x, pos.y, pos.z, 1
        );
    }

    public view(pos: Vec3, dir: Vec3, up: Vec3): Mat4 {
        const xAxis: Vec3 = Vec3.Cross(up, dir).normalize();
        const yAxis: Vec3 = Vec3.Cross(dir, xAxis);
        const posI: Vec3 = new Vec3(
            -(xAxis.x * pos.x + xAxis.y * pos.y + xAxis.z * pos.z),
            -(yAxis.x * pos.x + yAxis.y * pos.y + yAxis.z * pos.z),
            -(dir.x * pos.x + dir.y * pos.y + dir.z * pos.z),
        );
        // prettier-ignore
        return this.set(
            xAxis.x, yAxis.x, dir.x, 0,
            xAxis.y, yAxis.y, dir.y, 0,
            xAxis.z, yAxis.z, dir.z, 0,
            posI.x, posI.y, posI.z, 1
        );
    }

    public perspective(
        fov: float,
        aspect: float,
        near: float,
        far: float,
    ): Mat4 {
        const f: float = Math.tan(Math.PI * 0.5 - 0.5 * fov);
        const rangeInv: float = 1.0 / (near - far);
        // prettier-ignore
        return this.set(
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, far * rangeInv, -1,
            0, 0, near * far * rangeInv, 0,
        );
    }

    public orthogonal(
        left: float,
        right: float,
        top: float,
        bottom: float,
        near: float,
        far: float,
    ): Mat4 {
        // prettier-ignore
        return this.set(
            2 / (right - left), 0, 0, 0,
            0, 2 / (top - bottom), 0, 0,
            0, 0, 1 / (near - far), 0,
            (right + left) / (left - right), (top + bottom) / (bottom - top), near / (near - far), 1,
        );
    }

    public copy(mat: Mat4): Mat4 {
        return this.set(...mat.values);
    }

    public store(target: FloatArray, offset: int = 0): Mat4 {
        target[offset] = this.values[0];
        target[offset + 1] = this.values[1];
        target[offset + 2] = this.values[2];
        target[offset + 3] = this.values[3];
        target[offset + 4] = this.values[4];
        target[offset + 5] = this.values[5];
        target[offset + 6] = this.values[6];
        target[offset + 7] = this.values[7];
        target[offset + 8] = this.values[8];
        target[offset + 9] = this.values[9];
        target[offset + 10] = this.values[10];
        target[offset + 11] = this.values[11];
        target[offset + 12] = this.values[12];
        target[offset + 13] = this.values[13];
        target[offset + 14] = this.values[14];
        target[offset + 15] = this.values[15];
        return this;
    }

    public clone(): Mat4 {
        return new Mat4(this.isFloat64).copy(this);
    }

    public static Aim(
        pos: Vec3,
        dir: Vec3,
        up: Vec3,
        isFloat64?: boolean,
    ): Mat4 {
        return new Mat4(isFloat64).aim(pos, dir, up);
    }

    public static View(
        pos: Vec3,
        dir: Vec3,
        up: Vec3,
        isFloat64?: boolean,
    ): Mat4 {
        return new Mat4(isFloat64).view(pos, dir, up);
    }

    public static Perspective(
        fov: float,
        aspect: float,
        near: float,
        far: float,
        isFloat64?: boolean,
    ): Mat4 {
        return new Mat4(isFloat64).perspective(fov, aspect, near, far);
    }

    public static Orthogonal(
        left: float,
        right: float,
        top: float,
        bottom: float,
        near: float,
        far: float,
        isFloat64?: boolean,
    ): Mat4 {
        return new Mat4(isFloat64).orthogonal(
            left,
            right,
            top,
            bottom,
            near,
            far,
        );
    }
}
