/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { Bytes4, EntityLimit, EntityLayout } from "../constants.js";
import { assert } from "../utilities/utils.js";
import { Nullable, int } from "../utils.type.js";
import { Entity } from "../components/Entity.js";
import { Renderer } from "../core/Renderer.js";
import {
    Bounding,
    BufferWrite,
    ClusterId,
    EntityChange,
    EntityIndex,
    GeometryKey,
} from "../core.type.js";

export class EntityHandler {
    private readonly renderer: Renderer;

    private readonly list: Entity[];
    private readonly changes: Map<EntityIndex, EntityChange>;
    public readonly arrayBuffer: ArrayBuffer;
    public readonly uIntData: Uint32Array;
    public readonly floatData: Float32Array;
    public buffer: Nullable<GPUBuffer>;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.list = [];
        this.changes = new Map<EntityIndex, EntityChange>();
        this.arrayBuffer = new ArrayBuffer(EntityLayout * EntityLimit * Bytes4);
        this.uIntData = new Uint32Array(this.arrayBuffer);
        this.floatData = new Float32Array(this.arrayBuffer);
        this.buffer = null;
    }

    public prepare(): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device);
        this.buffer = device.createBuffer({
            label: "entity-buffer",
            size: this.arrayBuffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        } as GPUBufferDescriptor);
    }

    public append(entity: Entity): EntityIndex {
        assert(this.renderer.isReady());
        assert(this.renderer.handlers.geometry.exists(entity.key));
        assert(this.list.length < EntityLimit);
        let index: EntityIndex = this.list.push(entity) - 1;
        this.registerChange(index, "NEW");
        return index;
    }

    public getRootId(key: GeometryKey): ClusterId {
        return this.renderer.handlers.geometry.getRootId(key);
    }

    public getBounding(key: GeometryKey): Bounding {
        return this.renderer.handlers.geometry.getBounding(key);
    }

    public getLeaveTrianglesCount(key: GeometryKey): int {
        return this.renderer.handlers.geometry.getLeaveTrianglesCount(key);
    }

    public count(): int {
        return this.list.length;
    }

    public countLeaveTriangles(): int {
        let result: int = 0;
        for (let i: int = 0; i < this.list.length; i++) {
            result += this.list[i].getLeaveTrianglesCount();
        }
        return result;
    }

    public registerChange(index: EntityIndex, change: EntityChange): void {
        if (this.changes.has(index)) {
            return;
        }
        this.changes.set(index, change);
    }

    public gpuSync(): void {
        if (this.changes.size === 0) {
            return;
        }
        const writes: BufferWrite[] = [];
        const changes: EntityIndex[] = this.getSortedChanges();
        for (let i: int = 0; i < changes.length; i++) {
            let index: EntityIndex = changes[i];
            let offset: int = index * EntityLayout;
            this.list[index].getPosition().store(this.floatData, offset + 0);
            this.uIntData[offset + 3] = this.list[index].getRootId();
            this.floatData[offset + 4] = this.list[index].getBounding().radius;
            this.compactWrites(writes, offset);
        }
        this.executeWrites(writes);
        this.changes.clear();
    }

    private getSortedChanges(): EntityIndex[] {
        return Array.from(this.changes, ([index, _change]) => index).sort(
            (a: EntityIndex, b: EntityIndex) => a - b,
        );
    }

    private compactWrites(writes: BufferWrite[], offset: int): void {
        if (writes.length === 0) {
            return this.pushWrite(writes, offset);
        }
        let previous: BufferWrite = writes[writes.length - 1];
        if (offset * Bytes4 - (previous.dataOffset + previous.size) === 0) {
            previous.size += EntityLayout * Bytes4;
            return;
        }
        this.pushWrite(writes, offset);
    }

    private pushWrite(writes: BufferWrite[], offset: int): void {
        writes.push({
            bufferOffset: offset * Bytes4,
            dataOffset: offset * Bytes4,
            size: EntityLayout * Bytes4,
        } as BufferWrite);
    }

    private executeWrites(writes: BufferWrite[]): void {
        const device: Nullable<GPUDevice> = this.renderer.device;
        assert(device && this.buffer);
        for (let i: int = 0; i < writes.length; i++) {
            let write: BufferWrite = writes[i];
            device.queue.writeBuffer(
                this.buffer,
                write.bufferOffset,
                this.arrayBuffer,
                write.dataOffset,
                write.size,
            );
        }
    }
}
