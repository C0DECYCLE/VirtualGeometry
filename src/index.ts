/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { Entity } from "./components/Entity.js";
import { Renderer } from "./components/Renderer.js";
import { float, int } from "./utils.type.js";

const renderer: Renderer = new Renderer();

const keys: string[] = ["dralow" /*, "bunlow", "suhigh"*/];
for (let i: int = 0; i < keys.length; i++) {
    await renderer.import(keys[i], `./resources/${keys[i]}.obj`);
}

await renderer.prepare();

const n: int = 10;
const o: float = (n - 1) / 2;
const s: float = 5.0;
for (let i: int = 0; i < n; i++) {
    for (let j: int = 0; j < n; j++) {
        const entity: Entity = new Entity(
            keys[Math.floor(Math.random() * keys.length)],
        );
        entity.setPosition((i - o) * s, 0, (j - o) * s);
        renderer.add(entity);
    }
}

renderer.run();

(window as any).renderer = renderer;

renderer.handlers.uniform.viewMode(1);
