/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, February 2024
 */

import { OBJParseResult, OBJParser } from "../components/OBJParser.js";
import {
    Bounding,
    ClusterGroup,
    EdgeIdentifier,
    GeometryId,
    GeometryKey,
} from "../core.type.js";
import { log, warn } from "../utilities/logger.js";
import { assert, dotit } from "../utilities/utils.js";
import { Nullable, Undefinable, float, int } from "../utils.type.js";
import { Cluster } from "./Cluster.js";
import { Count } from "./Count.js";
import { Clustering } from "./Clustering.js";
import { Grouping } from "./Grouping.js";
import { Simplify } from "./Simplify.js";
import { Triangle } from "./Triangle.js";
import { Vertex } from "./Vertex.js";
import { Edge } from "./Edge.js";
import { Vec3 } from "../utilities/Vec3.js";

export class Virtual {
    public readonly id: GeometryId;
    public readonly key: GeometryKey;
    public bounding: Nullable<Bounding>;

    public readonly vertices: Vertex[];
    public readonly clusters: Cluster[];

    public leaveTrianglesCount: Undefinable<int>;

    public constructor(count: Count, key: GeometryKey, base: OBJParseResult) {
        this.id = count.registerGeometry();
        this.key = key;
        this.bounding = null;
        this.vertices = [];
        this.clusters = [];
        this.leaveTrianglesCount = undefined;
        if (base.vertices && base.verticesCount) {
            this.baseObj(count, base);
            this.flush();
        } else {
            //...
            //here virtual prebuild version
        }
    }

    private baseObj(count: Count, obj: OBJParseResult): void {
        if (obj.vertexColors) {
            warn("Virtual: Vertex-Colors not supported yet.");
        }
        const pre: float = performance.now();
        this.extractVertices(count, obj);
        // prettier-ignore
        const triangles: Triangle[] = this.extractTrianglesEdges(count, obj, this.vertices);
        this.leaveTrianglesCount = triangles.length;
        // prettier-ignore
        const leaves: Cluster[] = Clustering.ClusterizeWithAdjacency(count, triangles);
        this.constructDAGAndTree(count, leaves);
        //export .virtual
        log(
            `Virtual: "${this.key}" (${dotit(
                this.clusters.length,
            )} clusters, ${dotit(performance.now() - pre)} ms)`,
        );
    }

    private extractVertices(count: Count, parse: OBJParseResult): void {
        const stride: int = parse.vertices.length / parse.verticesCount;
        //assert(stride === VertexLayout);
        const min: Vec3 = new Vec3();
        const max: Vec3 = new Vec3();
        for (let i: int = 0; i < parse.verticesCount; i++) {
            const vertex: Vertex = new Vertex(
                count,
                parse.vertices.slice(i * stride, (i + 1) * stride),
            );
            this.vertices.push(vertex);
            min.x = Math.min(min.x, vertex.position.x);
            min.y = Math.min(min.y, vertex.position.y);
            min.z = Math.min(min.z, vertex.position.z);
            max.x = Math.max(max.x, vertex.position.x);
            max.y = Math.max(max.y, vertex.position.y);
            max.z = Math.max(max.z, vertex.position.z);
        }
        const center: Vec3 = min.clone().add(max).scale(0.5);
        this.bounding = {
            min: min,
            max: max,
            center: center,
            radius: min.clone().sub(center).length(),
        } as Bounding;
    }

    private extractTrianglesEdges(
        count: Count,
        parse: OBJParseResult,
        vertices: Vertex[],
    ): Triangle[] {
        assert(parse.indices && parse.indicesCount);
        const triangles: Triangle[] = [];
        const stride: int = 3;
        const n: int = parse.indicesCount / stride;
        const edges: Map<EdgeIdentifier, Edge> = new Map<
            EdgeIdentifier,
            Edge
        >();
        for (let i: int = 0; i < n; i++) {
            const triangle: Triangle = new Triangle(count, [
                vertices[parse.indices[i * stride + 0]],
                vertices[parse.indices[i * stride + 1]],
                vertices[parse.indices[i * stride + 2]],
            ]);
            triangles.push(triangle);
            triangle.registerEdgesToMap(edges);
        }
        return triangles;
    }

    private constructDAGAndTree(count: Count, leaves: Cluster[]): void {
        this.clusters.push(...leaves);
        let previous: Cluster[] = leaves;
        while (previous.length > 1) {
            const groups: ClusterGroup[] = Grouping.Group(previous);
            const next: Cluster[] = [];
            for (let i: int = 0; i < groups.length; i++) {
                const children: ClusterGroup = groups[i];
                const { triangles, edges } = Grouping.Merge(count, children);
                Simplify.Simplify(count, this.vertices, triangles, edges);
                const parents: ClusterGroup =
                    Clustering.ClusterizeWithoutAdjacency(count, triangles);
                this.setChildrenParents(children, parents);
                next.push(...parents);
            }
            this.clusters.push(...next);
            previous = next;
        }
        const root: Cluster = this.clusters[this.clusters.length - 1];
        this.extractTree(root);
    }

    private setChildrenParents(
        children: ClusterGroup,
        parents: ClusterGroup,
    ): void {
        let maxChildError: float = 0;
        for (let i: int = 0; i < children.length; i++) {
            maxChildError = Math.max(maxChildError, children[i].error);
        }
        let maxParentError: float = 0;
        for (let i: int = 0; i < parents.length; i++) {
            maxParentError = Math.max(maxParentError, parents[i].error);
        }
        const parentError: float = maxChildError + maxParentError;
        for (let i: int = 0; i < children.length; i++) {
            const child: Cluster = children[i];
            assert(!child.parentError);
            child.parentError = parentError;
        }
        const treeChildren: Cluster[] = [...children];
        for (let i: int = 0; i < parents.length; i++) {
            const parent: Cluster = parents[i];
            parent.error = parentError;
            assert(!parent.childrenLength && !parent.tree);
            parent.childrenLength = children.length;
            parent.tree = treeChildren;
        }
    }

    private extractTree(cluster: Cluster): void {
        if (!cluster.childrenLength) {
            //null or 0
            return;
        }
        assert(cluster.tree);
        const target: int = Math.min(cluster.childrenLength, 2);
        if (cluster.tree.length !== target) {
            cluster.tree = cluster.tree.splice(0, target);
        }
        for (let i: int = 0; i < cluster.tree.length; i++) {
            this.extractTree(cluster.tree[i]);
        }
    }

    private flush(): void {
        for (let i: int = 0; i < this.clusters.length; i++) {
            this.clusters[i].flush();
        }
    }

    public static async Create(
        count: Count,
        key: GeometryKey,
        path: string,
    ): Promise<Virtual> {
        const prebuild: boolean = await Virtual.Exist(`${path}.virtual`);
        if (prebuild) {
            throw new Error("Virtual: Not Implemented.");
            //...
            //load .virtual string and parse
            //new Virtual(count, key, virtual);
        }
        const obj: OBJParseResult = OBJParser.Standard.parse(
            await Virtual.Load(path),
            true,
        );
        return new Virtual(count, key, obj);
    }

    private static async Exist(path: string): Promise<boolean> {
        return await fetch(path, { method: "HEAD" }).then(
            async (response: Response) => response.ok,
        );
    }

    private static async Load(path: string): Promise<string> {
        return await fetch(path).then(
            async (response: Response) => await response.text(),
        );
    }
}
