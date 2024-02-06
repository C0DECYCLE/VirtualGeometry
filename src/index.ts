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

//CORE:
//get working with all models
//only based on object distance not cluster distance!
//error random - bad?
//multiple instances

//OPTIMIZE:
//improve code, refactor reduce memory and redudant stuff
//evaluate each cluster currently
//  -> should do tree based not evaluate children with persistant threads and atomic queue
//  -> tree based on parenterror? for deciding to traverse children?
//cluster frustum culling
//instance frustum culling

//freeze mode for debug
