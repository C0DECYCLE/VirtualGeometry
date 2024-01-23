/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { OBJParseResult, OBJParser } from "../components/OBJParser";
import { GeometryHandlerCount } from "./Counts";
import { Geometry } from "./Geometry.js";

export class GeometryHandler {
    public readonly geometries: Map<string, Geometry>;

    public readonly count: GeometryHandlerCount;

    public constructor() {
        this.geometries = new Map<string, Geometry>();
        this.count = new GeometryHandlerCount(0, 0, 0, 0);
    }

    public async import(key: string, path: string): Promise<void> {
        const parse: OBJParseResult = OBJParser.Standard.parse(
            await this.loadText(path),
            true,
        );
        this.geometries.set(key, new Geometry(this.count, parse));
    }

    private async loadText(path: string): Promise<string> {
        return await fetch(path).then(
            async (response: Response) => await response.text(),
        );
    }
}
