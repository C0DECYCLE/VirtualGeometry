/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { Renderer } from "./components/Renderer.js";

const renderer: Renderer = new Renderer();
await renderer.importGeometry("suzanne", "./resources/suhigh.obj");
await renderer.initialize();
//const object: RenderObject = renderer.add(new Vec3(0, 0, 0), "suzanne");
renderer.run();

//make plan with classes,
//revert renderer back into less big function or leave like this start with geometry preprocessing,
//use current as debugging and if acceptable remove everything and start clean;

//about the meshlet generation: bounding ones were realy slow and memory heavy with bad results
//plan new one similar to meshoptimizer library and greedy, and implement
//first build triangle data with adjacent calculations
