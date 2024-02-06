/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { ShaderPaths } from "../constants.js";
import { Nullable } from "../utils.type.js";

export class ShaderHandler {
    public evaluationModule: Nullable<GPUShaderModule>;
    public drawModule: Nullable<GPUShaderModule>;

    public constructor() {
        this.evaluationModule = null;
        this.drawModule = null;
    }

    public async prepare(device: GPUDevice): Promise<void> {
        this.evaluationModule = await this.requestModule(
            device,
            ShaderPaths.evaluation,
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
