/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { OBJParseResult } from "../components/OBJParser";
import { Vec3 } from "../utilities/Vec3";
import { int } from "../utils.type";

export class Geometry {
    public readonly vertices: Vec3[];
    public constructor(parse: OBJParseResult) {
        if (parse.vertexColors) {
            throw new Error("Geometry: Vertex-Colors not supported yet.");
        }
        this.vertices = this.extractVerices(parse);
    }

    private extractVertices(): Vec3[];
}
