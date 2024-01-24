/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { TextureFormats } from "../constants.js";
import { Nullable } from "../utils.type.js";

export class TextureHandler {
    public textures: Nullable<{ depth: GPUTexture }>;
    public views: Nullable<{ depth: GPUTextureView }>;

    public constructor() {
        this.textures = null;
        this.views = null;
    }

    public prepare(device: GPUDevice, canvas: HTMLCanvasElement): void {
        this.textures = {
            depth: this.createDepthTexture(device, canvas),
        };
        this.views = {
            depth: this.textures.depth.createView(),
        };
    }

    private createDepthTexture(
        device: GPUDevice,
        canvas: HTMLCanvasElement,
    ): GPUTexture {
        return device.createTexture({
            label: "depth-texture",
            size: [canvas.width, canvas.height],
            format: TextureFormats.depth,
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING,
        } as GPUTextureDescriptor);
    }
}
