/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { EntityLimit, EntityStride } from "../constants.js";
import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { Entity } from "./Entity.js";
import { Renderer } from "./Renderer.js";

export class EntityHandler {
    private readonly renderer: Renderer;

    private readonly list: Entity[];
    private readonly data: Float32Array;
    public buffer: Nullable<GPUBuffer>;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.list = [];
        this.data = new Float32Array(EntityStride * EntityLimit);
        this.buffer = null;
    }

    public prepare(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.buffer = device.createBuffer({
            label: "entity-buffer",
            size: this.data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
    }

    public add(entity: Entity): Entity {
        assert(this.renderer.isReady());
        assert(this.renderer.geometryHandler.exists(entity.key));
        assert(!this.list.includes(entity));
        this.list.push(entity);
        this.synchronize(); //entity doesnt get updated!
        return entity;
    }

    public synchronize(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device && this.buffer);
        for (let i: int = 0; i < this.list.length; i++) {
            this.list[i].position.store(this.data, i * EntityStride);
        }
        device.queue.writeBuffer(this.buffer, 0, this.data);
    }
}

//here smart look notes
