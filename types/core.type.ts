/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { Vec3 } from "./utilities/Vec3";
import { int } from "./utils.type";

export type GeometryId = int;

export type GeometryKey = string;

/*
export type GeometryExport = {
    vertices: VertexExport[];
    clusters: ClusterExport[];
};
*/

export type ClusterId = int;

export type ClusterCenter = {
    sum: Vec3;
    n: int;
    center: Vec3;
};

export type ClusterBounds = {
    min: Vec3;
    max: Vec3;
};

/*
export type ClusterExport = {
    triangles: TriangleExport[];
    clusters: ClusterExport[];
};
*/

export type TriangleId = int;

export type EdgeIdentifier = string; // VertexId-VertexId // first lower !

export type VertexId = int;

//export type VertexExport = [float, float, float];

export type EntityIndex = int;

export type EntityChange = "NEW" | "UPDATE"; // | "DELETE"

export type bytes = int;

export type BufferWrite = {
    bufferOffset: bytes;
    dataOffset: int;
    size: int;
};
