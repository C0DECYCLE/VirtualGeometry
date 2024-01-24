/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { Entity } from "./core/Entity.js";
import { Renderer } from "./core/Renderer.js";
import { Vec3 } from "./utilities/Vec3.js";
import { log } from "./utilities/logger.js";

const renderer: Renderer = new Renderer();
await renderer.import("test", "./resources/suzanne.obj");
await renderer.prepare();
renderer.add(new Entity(new Vec3(0, 0, 0), "test"));
log(renderer);
renderer.run();

//clean up, improve, make plan for better structure based on notes
