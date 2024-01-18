/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, December 2023
 */

import { float } from "../../types/utils.type.js";
import { Mat4 } from "../utilities/Mat4.js";
import { Vec3 } from "../utilities/Vec3.js";
import { toRadian } from "../utilities/utils.js";

export class Camera {
    private readonly view: Mat4;
    private readonly projection: Mat4;
    private readonly viewProjection: Mat4;

    public readonly position: Vec3;
    public readonly direction: Vec3;
    private readonly up: Vec3;

    public constructor(aspect: float, far: float) {
        this.view = new Mat4();
        this.projection = Mat4.Perspective(60 * toRadian, aspect, 0.1, far);
        this.viewProjection = new Mat4();
        this.position = new Vec3(0, 0, 0);
        this.direction = new Vec3(0, 0, 1);
        this.up = new Vec3(0, 1, 0);
    }

    public update(): Mat4 {
        this.view.view(this.position, this.direction, this.up);
        return this.viewProjection.multiply(this.view, this.projection);
    }
}
