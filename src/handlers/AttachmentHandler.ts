/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { assert } from "../utilities/utils.js";
import { Nullable } from "../utils.type.js";
import { Renderer } from "../components/Renderer.js";
import { TextureHandler } from "./TextureHandler.js";

export class AttachmentHandler {
    private readonly renderer: Renderer;

    public color: Nullable<GPURenderPassColorAttachment>;
    public depthStencil: Nullable<GPURenderPassDepthStencilAttachment>;

    public constructor(renderer: Renderer) {
        this.renderer = renderer;
        this.color = null;
        this.depthStencil = null;
    }

    public prepare(): void {
        this.color = this.createColor();
        this.depthStencil = this.createDepthStencil();
    }

    private createColor(): GPURenderPassColorAttachment {
        return {
            label: "color-attachment",
            view: this.renderer.requestContextView(),
            clearValue: [0.3, 0.3, 0.3, 1],
            loadOp: "clear",
            storeOp: "store",
        } as GPURenderPassColorAttachment;
    }

    private createDepthStencil(): GPURenderPassDepthStencilAttachment {
        const textures: TextureHandler = this.renderer.handlers.texture;
        assert(textures.views);
        return {
            label: "depth-stencil-attachment",
            view: textures.views.depth,
            depthClearValue: 1,
            depthLoadOp: "clear",
            depthStoreOp: "store",
        } as GPURenderPassDepthStencilAttachment;
    }

    public update(): void {
        assert(this.color);
        this.color.view = this.renderer.requestContextView();
    }
}
