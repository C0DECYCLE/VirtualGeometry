/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, December 2023
 */

import { int } from "../../types/utilities/utils.type.js";
import { OBJParseResult, OBJParser } from "../OBJParser.js";
import { Vec3 } from "../utilities/Vec3.js";

export const byteSize: int = 4;

export function createCanvas(): HTMLCanvasElement {
    const canvas: HTMLCanvasElement = document.createElement("canvas");
    canvas.width = document.body.clientWidth * devicePixelRatio;
    canvas.height = document.body.clientHeight * devicePixelRatio;
    canvas.style.position = "absolute";
    canvas.style.top = "0px";
    canvas.style.left = "0px";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    document.body.appendChild(canvas);
    return canvas;
}

export async function loadText(path: string): Promise<string> {
    return await fetch(path).then(
        async (response: Response) => await response.text(),
    );
}

export async function loadOBJ(
    path: string,
    color?: Vec3,
): Promise<OBJParseResult> {
    return OBJParser.Standard.parse(await loadText(path), true, color);
}
