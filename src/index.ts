/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { Entity } from "./core/Entity.js";
import { Renderer } from "./core/Renderer.js";
import { log } from "./utilities/logger.js";

const renderer: Renderer = new Renderer();
await renderer.import("test", "./resources/bunlow.obj");
await renderer.prepare();
renderer.add(new Entity("test"));
log(renderer);
renderer.run();

//old -> new clustering:
//suzanne: 23ms -> 16ms
//suhigh: 170ms -> 42ms
//bunlow: 2300ms -> 219ms
//bunmid: 9000ms -> 766ms
//bunny: 37000ms -> 2612ms
