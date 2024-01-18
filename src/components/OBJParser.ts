/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { int, float, Nullable } from "../utilities/utils.type.js";
import { clear } from "../utilities/utils.js";
import { Vec3 } from "../utilities/Vec3.js";

type OBJVertex = [float, float, float, float?, float?, float?];
export type OBJParseResult = {
    vertices: Float32Array;
    verticesCount: int;
    vertexColors: boolean;
    indices?: Uint32Array;
    indicesCount?: int;
};

export class OBJParser {
    public static readonly Standard: OBJParser = new OBJParser();

    private readonly cache: OBJVertex[];
    private readonly vertexCache: float[];
    private vertexColors: boolean;
    private readonly indexCache: int[];

    public constructor() {
        this.cache = [];
        this.vertexCache = [];
        this.vertexColors = false;
        this.indexCache = [];
    }

    public parse(
        raw: string,
        indexed: boolean = false,
        color?: Vec3,
    ): OBJParseResult {
        this.reset();
        const regExp: RegExp = /(\w*)(?: )*(.*)/;
        const lines: string[] = raw.split("\n");
        for (let i: int = 0; i < lines.length; ++i) {
            this.parseLine(regExp, lines[i].trim(), indexed, color);
        }
        const result: OBJParseResult = {
            vertices: new Float32Array(this.vertexCache),
            verticesCount:
                this.vertexCache.length / (this.vertexColors ? 8 : 4),
            vertexColors: this.vertexColors,
        } as OBJParseResult;
        if (indexed) {
            result.indices = new Uint32Array(this.indexCache);
            result.indicesCount = this.indexCache.length;
        }
        this.reset();
        return result;
    }

    private parseLine(
        regExp: RegExp,
        line: string,
        indexed: boolean,
        color?: Vec3,
    ): void {
        const m: Nullable<RegExpExecArray> = regExp.exec(line);
        if (line === "" || line.startsWith("#") || !m) {
            return;
        }
        const parts: string[] = line.split(/\s+/).slice(1);
        switch (m[1]) {
            case "v":
                return this.keywordV(parts, indexed, color);
            case "f":
                return this.keywordF(parts, indexed);
        }
    }

    private keywordV(parts: string[], indexed: boolean, color?: Vec3): void {
        if (parts.length < 3) {
            throw new Error(`ObjParser: Obj file missing vertex part.`);
        }
        const x: float = parseFloat(parts[0]);
        const y: float = parseFloat(parts[1]);
        const z: float = parseFloat(parts[2]);
        const vertex: OBJVertex = [x, y, z];
        if (parts.length === 6 || color) {
            this.vertexColors = true;
            const r: float = color ? color.x : parseFloat(parts[3]);
            const g: float = color ? color.y : parseFloat(parts[4]);
            const b: float = color ? color.z : parseFloat(parts[5]);
            vertex.push(r, g, b);
        }
        this.cache.push(vertex);
        if (!indexed) {
            return;
        }
        this.registerVertex(vertex);
    }

    private keywordF(parts: string[], indexed: boolean): void {
        const a: int = parseInt(parts[0]) - 1;
        const b: int = parseInt(parts[1]) - 1;
        const c: int = parseInt(parts[2]) - 1;
        if (!indexed) {
            this.registerVertex(this.cache[a]);
            this.registerVertex(this.cache[b]);
            this.registerVertex(this.cache[c]);
            return;
        }
        this.indexCache.push(a, b, c);
    }

    private registerVertex(vertex: OBJVertex): void {
        this.vertexCache.push(vertex[0], vertex[1], vertex[2], 0);
        if (vertex[3] && vertex[4] && vertex[5]) {
            this.vertexCache.push(vertex[3], vertex[4], vertex[5], 0);
        }
    }

    public reset(): void {
        clear(this.cache);
        clear(this.vertexCache);
        this.vertexColors = false;
        clear(this.indexCache);
    }

    public destroy(): void {
        this.reset();
    }
}
