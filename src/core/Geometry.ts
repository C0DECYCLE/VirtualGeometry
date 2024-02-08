/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { OBJParseResult } from "../components/OBJParser.js";
import { GeometryId, GeometryKey } from "../core.type.js";
import { log, warn } from "../utilities/logger.js";
import { assert, dotit } from "../utilities/utils.js";
import { float, int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";
import { Count } from "./Count.js";
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

        //this.export();

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
        const clusterize = GeometryClustering.ClusterizeWithoutAdjacency;
        const hirarchy: Cluster[] = [...leaves];
        let previous: Cluster[] = leaves;
        while (previous.length > 1) {
            const groups: Cluster[][] = GeometryGrouping.Group(previous);
            const next: Cluster[] = [];
            for (let i: int = 0; i < groups.length; i++) {
                const children: Cluster[] = groups[i];
                const { triangles, edges } = merge(count, children);
                simplify(count, this.vertices, triangles, edges);
                const parents: Cluster[] = clusterize(count, triangles);
                this.setChildrenParents(children, parents);
                next.push(...parents);
            }
            hirarchy.push(...next);
            previous = next;
        }
        this.extractTree(previous[0]);

        let treeNodes: int = 0;
        const nodeSet: Set<Cluster> = new Set<Cluster>();
        function walkTree(node: Cluster): void {
            treeNodes++;
            nodeSet.add(node);
            if (!node.treeChildren) {
                return;
            }
            assert(node.treeChildren.length <= 2);
            for (let i: int = 0; i < node.treeChildren.length; i++) {
                walkTree(node.treeChildren[i]);
            }
        }
        walkTree(previous[0]);
        log(treeNodes, nodeSet);

        return hirarchy;
    }

    private setChildrenParents(children: Cluster[], parents: Cluster[]): void {
        let maxChildrenError: float = 0;
        for (let i: int = 0; i < children.length; i++) {
            const child: Cluster = children[i];
            maxChildrenError = Math.max(maxChildrenError, child.error);
            assert(!child.parents);
            child.parents = parents;
        }
        const parentError: float = maxChildrenError + Math.random() * 0.2 + 0.1;
        const treeChildren: Cluster[] = [...children];
        for (let i: int = 0; i < parents.length; i++) {
            const parent: Cluster = parents[i];
            parent.error = parentError;
            assert(!parent.children && !parent.treeChildren);
            parent.children = children;
            parent.treeChildren = treeChildren;
        }
    }

    private extractTree(cluster: Cluster): void {
        if (!cluster.children) {
            return;
        }
        assert(cluster.treeChildren);
        const target: int = Math.min(cluster.children.length, 2);
        if (cluster.treeChildren.length !== target) {
            cluster.treeChildren = cluster.treeChildren.splice(0, target);
        }
        for (let i: int = 0; i < cluster.treeChildren.length; i++) {
            this.extractTree(cluster.treeChildren[i]);
        }
    }

    /*
    private export(): void {
        const result: 
    }
    */
}
