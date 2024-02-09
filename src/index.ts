/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { Entity } from "./components/Entity.js";
import { Renderer } from "./components/Renderer.js";

const renderer: Renderer = new Renderer();
await renderer.import("test", "./resources/bunlow.obj");
await renderer.prepare();
renderer.add(new Entity("test"));
renderer.run();

(window as any).renderer = renderer;

renderer.handlers.uniform.viewMode(1);
