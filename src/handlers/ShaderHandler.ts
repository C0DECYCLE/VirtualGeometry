/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ShaderPaths } from "../constants.js";
import { Nullable } from "../utils.type.js";

export class ShaderHandler {
    public clusterModule: Nullable<GPUShaderModule>;
    public drawModule: Nullable<GPUShaderModule>;

    public constructor() {
        this.clusterModule = null;
        this.drawModule = null;
    }

    public async prepare(device: GPUDevice): Promise<void> {
        this.clusterModule = await this.requestModule(
            device,
            ShaderPaths.cluster,
        );
        this.drawModule = await this.requestModule(device, ShaderPaths.draw);
    }

    private async requestModule(
        device: GPUDevice,
        path: string,
    ): Promise<GPUShaderModule> {
        return device.createShaderModule({
            label: `shader-module-${path}`,
            code: await this.loadText(path),
        } as GPUShaderModuleDescriptor);
    }

    private async loadText(path: string): Promise<string> {
        return await fetch(path).then(
            async (response: Response) => await response.text(),
        );
    }
}
