/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { Entity } from "./core/Entity.js";
import { Renderer } from "./core/Renderer.js";

const renderer: Renderer = new Renderer();
await renderer.import("test", "./resources/suhigh.obj");
await renderer.prepare();
renderer.add(new Entity("test"));
renderer.run();

(window as any).renderer = renderer;
