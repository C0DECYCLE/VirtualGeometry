/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { Entity } from "./core/Entity.js";
import { Renderer } from "./core/Renderer.js";
import { Vec3 } from "./utilities/Vec3.js";
import { log } from "./utilities/logger.js";

const renderer: Renderer = new Renderer();
await renderer.geometryHandler.import("test", "./resources/bunlow.obj");
const entity: Entity = new Entity(new Vec3(0, 0, 0), "test");
await renderer.prepare();
renderer.entityHandler.add(entity);
log(renderer, entity);
renderer.run();

//clean up, improve, make plan for better structure based on notes
