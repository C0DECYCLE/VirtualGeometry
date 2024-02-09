/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { Cluster } from "./generator/Cluster";
import { Nullable, int } from "./utils.type";

export type Flushable<T> = Nullable<T>;

export type GeometryId = int;

export type GeometryKey = string;

/*
export type GeometryExport = {
    vertices: VertexExport[];
    clusters: ClusterExport[];
};
*/

export type ClusterId = int;

export type ClusterGroup = Cluster[];

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
