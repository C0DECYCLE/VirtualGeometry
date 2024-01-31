/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, January 2024
 */

import { OBJParseResult } from "../components/OBJParser.js";
import { ClusterGroupingLimit } from "../constants.js";
import { EdgeIdentifier, GeometryId, GeometryKey } from "../core.type.js";
import { log, warn } from "../utilities/logger.js";
import { assert, dotit, swapRemove } from "../utilities/utils.js";
import { Nullable, Undefinable, float, int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";
import { Count } from "./Count.js";
import { Edge } from "./Edge.js";
import { GeometryClustering } from "./GeometryClustering.js";
import { GeometryGrouping } from "./GeometryGrouping.js";
import { GeometryHelper } from "./GeometryHelper.js";
import { GeometrySimplify } from "./GeometrySimplify.js";
import { Triangle } from "./Triangle.js";
import { Vertex } from "./Vertex.js";

export class Geometry {
    public readonly key: GeometryKey;
    public readonly id: GeometryId;

    public readonly vertices: Vertex[];
    public readonly clusters: Cluster[];

    public constructor(count: Count, key: GeometryKey, parse: OBJParseResult) {
        if (parse.vertexColors) {
            warn("Geometry: Vertex-Colors not supported yet.");
        }
        this.key = key;
        this.id = count.registerGeometry();
        const extract = GeometryHelper.ExtractTrianglesEdges;
        const clusterize = GeometryClustering.Clusterize;

        const pre: float = performance.now(); //

        this.vertices = GeometryHelper.ExtractVertices(count, parse);
        const triangles: Triangle[] = extract(count, parse, this.vertices);
        const leaves: Cluster[] = clusterize(count, triangles);
        this.clusters = this.buildHirarchy(count, leaves);

        //
        log(
            this.key,
            ":",
            dotit(this.clusters.length),
            "clusters",
            "in",
            dotit(performance.now() - pre),
            "ms",
        );
    }

    private buildHirarchy(count: Count, leaves: Cluster[]): Cluster[] {
        const merge = GeometryHelper.MergeClusters;
        const simplify = GeometrySimplify.Simplify;
        const hirarchy: Cluster[] = [...leaves];

        /*
        log(leaves.length); //

        let previous: Cluster[] = leaves;
        while (previous.length > 1) {
            const groups: Cluster[][] = GeometryGrouping.Group(previous);
            const next: Cluster[] = [];
            for (let i: int = 0; i < groups.length; i++) {
                //
                try {
                    const { triangles, edges } = merge(count, groups[i]);
                    simplify(count, this.vertices, triangles, edges);
                    // prettier-ignore
                    next.push(...GeometryClustering.Clusterize(count, triangles));
                    //
                } catch (e) {
                    hirarchy.push(...next); //
                    log(e, next.length); //
                    return hirarchy; //
                }
            }
            hirarchy.push(...next);
            previous = next;

            log(next.length); //
        }
        */
        return hirarchy;
    }
}
