/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { Bytes4, UniformsLayout } from "../constants.js";
import { Mat4 } from "../utilities/Mat4.js";
import { Vec3 } from "../utilities/Vec3.js";
import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";

export class UniformHandler {
    public readonly arrayBuffer: ArrayBuffer;
    public readonly uIntData: Uint32Array;
    public readonly floatData: Float32Array;
    public buffer: Nullable<GPUBuffer>;

    public constructor() {
        this.arrayBuffer = new ArrayBuffer(UniformsLayout * Bytes4);
        this.uIntData = new Uint32Array(this.arrayBuffer);
        this.floatData = new Float32Array(this.arrayBuffer);
        this.buffer = null;
    }

    public prepare(device: GPUDevice): void {
        this.buffer = device.createBuffer({
            label: "uniforms-buffer",
            size: this.arrayBuffer.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
    }

    public viewProjection(matrix: Mat4): void {
        matrix.store(this.floatData, 0);
    }

    public viewMode(mode: int): void {
        this.uIntData[16] = mode;
    }

    public resolution(width: int, height: int): void {
        this.floatData[18] = width;
        this.floatData[19] = height;
    }

    public cameraPosition(position: Vec3): void {
        position.store(this.floatData, 20);
    }

    public synchronize(device: GPUDevice): void {
        assert(this.buffer);
        device.queue.writeBuffer(this.buffer, 0, this.arrayBuffer);
    }
}
