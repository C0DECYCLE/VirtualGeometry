/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { Entity } from "./core/Entity.js";
import { Renderer } from "./core/Renderer.js";
import { log } from "./utilities/logger.js";

const renderer: Renderer = new Renderer();
await renderer.import("test", "./resources/suzanne.obj");
await renderer.prepare();
renderer.add(new Entity("test"));
log(renderer);
renderer.run();
