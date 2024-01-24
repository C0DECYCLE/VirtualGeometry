/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { UniformsLayout } from "../constants.js";
import { assert } from "../utilities/utils.js";
import { Nullable } from "../utils.type.js";

export class UniformHandler {
    public readonly data: Float32Array;
    public buffer: Nullable<GPUBuffer>;

    public constructor() {
        this.data = new Float32Array(UniformsLayout);
        this.buffer = null;
    }

    public prepare(device: GPUDevice): void {
        this.buffer = device.createBuffer({
            label: "uniforms-buffer",
            size: this.data.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
    }

    public synchronize(device: GPUDevice): void {
        assert(this.buffer);
        device.queue.writeBuffer(this.buffer, 0, this.data);
    }
}
