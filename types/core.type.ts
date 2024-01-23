/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { Vec3 } from "./utilities/Vec3";
import { float, int } from "./utils.type";

export type GeometryId = int;

export type GeometryKey = string;

export type ClusterId = int;

export type ClusterCenter = {
    sum: Vec3;
    n: int;
    center: Vec3;
};

export type ClusterBounds = {
    center: Vec3;
    radius: float;
};

export type TriangleId = int;

export type VertexId = int;
