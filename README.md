# Virtual Geometry

## Resources

### Nanite

-   https://www.youtube.com/watch?v=eviSykqSUUw
-   https://www.youtube.com/watch?v=NRnj_lnpORU&t=5027s
-   https://www.youtube.com/watch?v=TMorJX3Nj6U&t=3982s

-   https://cs418.cs.illinois.edu/website/text/nanite.html
-   https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf
-   https://advances.realtimerendering.com/s2015/aaltonenhaar_siggraph2015_combined_final_footer_220dpi.pdf
-   https://computergraphics.stackexchange.com/questions/11060/what-are-mesh-clusters-hierarchical-cluster-culling-with-lod-triangle-c
-   https://www.elopezr.com/a-macro-view-of-nanite/
-   https://www.notion.so/Brief-Analysis-of-Nanite-94be60f292434ba3ae62fa4bcf7d9379
-   https://gamedev.stackexchange.com/questions/198454/how-does-unreal-engine-5s-nanite-work

### Other

-   https://discourse.threejs.org/t/virtually-geometric/28420/40
-   https://jcgt.org/published/0012/02/01/
-   https://www.youtube.com/watch?v=7JEHPvSGaX8
-   https://github.com/OmarShehata/webgpu-compute-rasterizer/blob/main/how-to-build-a-compute-rasterizer.md
-   https://graphics.stanford.edu/courses/cs468-10-fall/LectureSlides/08_Simplification.pdf
-   https://www.ri.cmu.edu/pub_files/pub2/garland_michael_1997_1/garland_michael_1997_1.pdf
-   https://blog.traverseresearch.nl/creating-a-directed-acyclic-graph-from-a-mesh-1329e57286e5

## Idea

### Instance

-   position (vec3, float32)
-   geometryId (unique globally)
-   instanceId (= index in buffer, unique globally)

### Cluster

-   max 128 tris (indexed?)
-   boudning info
-   clusterId (unique geometry-wide, position in linear? (to find parent, siblings, children?))
-   (store tree in one buffer and all the cluster data in another? or same)

### Compiletime

1.  import and parse geometry
2.  divde into leave clusters, group, merge, simplify, split into new clusters
    upwards to build dag, encode in linear format later custom tree based on parenterror
3.  build attributes for all clusters
    for each instance (=root cluster) build its info

### Runtime

0.  (first sync cpu-gpu object changes?)
1.  for each instance cull in compute (reduce as much as possible) (go trough instance  
    buffer and write survived ids into new buffer)
2.  for each surviving instance and its geometry tree determine the lod level and cull
    (reduce as much as possible)
3.  add all still remaining clusters (aka its instance id and cluster id) to a (draw) buffer
4.  issue one non indexed draw call on that buffer (vertexCount = 128 tris \* 3,  
    instanceCount = number of clusters)
    and the draw shader fetches the correct data via the instance_index and vertex_index:
    use the instance_index to find the current cluster data (instanceId of object and clusterId inside object)
    then u can get the object position via the instanceId into the instance buffer
    and you can get the index of the vertex via the clusterId + vertex_index into the index buffer which will point to the vertex buffer

### Notes

-   while import check for limits, max number of indexes (half of 32bit ? because of)
-   use float16 somewhere?
-   do all dispatchWorkgroupIndirect amazing!!!
-   build softare rasterizer? for small clusters? triangles?
-   build occlusing culling? two pass hirarchical z buffer
-   build non lod cracks? dag graph partitiong
-   multiple shading modes for debug
-   check which lod to show by calculating its screen size (use bounding sphere info to project to points onto screen)
-   record all object property changes and write them chunk vise all together before draw process, every object holds its index in the instance buffer, if deleted the place gets registered in the cpu as free and id a new gets created it gets filled, this would mean holes🤔 not bad because compute shader can skip them? or should they be filled up by swapping last one in? anyway egal which case it requires a cpu-gpu memory write. how do they work around buffer limits? wouldnt that mean a general instance limit?
-   make debug mode for freezing and also debug shadings (diffrenet ids diffrent colors etc)
-   vertex, triangle, cluster, geometry class make abstraction and id types, then make good renderer and just schedule the normal / all clusters
-   atomic also on arrays!
-   persistent threads: simply block termination of gpu thread with while loop until global atomics guarded work queue is empty!

### TODO

-   Entity deletion
-   improve clustering to reduce sprinkeld in triangles between stuff
-   manual pipeline bind group layouts

### Clustering

0.  all vertices into array and triangles with links to vertices into arry
1.  compute all adjacent triangles for each triangle. adjacent if they share min 2 vertices
    or alternativly if they share a edge (register all edges from triangles without duplicates and register in edge which triangle its a part of, if edge is part of 2 triangles write to those triangles that they are adjacent) or triangle register edges and later find adjacent by looking which triangles my edges also have
2.  put all unclustered triangles into a queue, start with a random triangle, while cluster
    is not full and there are unclustered triangles left add adjacent triangles of the triangles already in the cluster to the cluster. if no more adjacent add one from the unused queue. in any case add the one next which the least increases the bounding sphere of the cluster. when finish with a cluster use one adjacent to the outside as the start for the next.

### Merge & Simplify

0.  merge by just merging the list of triangles
1.  first construct map of all vertices and triangles by id. then register all edges of all
    triangles into map with key based on the ids of the vertices. write into each edge which triangles is it a part off, if edge is part of only 1 triangle its a border edge. (mark the vertices of border edges as border vertices.)? "lock" aka dont change position and deletion of border vertices.
2.  simplification only inside of mesh meaning dont simplify a border edge. simplification  
    happens by edge collapse meaning merge the vertices of the edge together and update the vertices edges triangle. merge them togther at the average position aka middle? middle is the thing about average vs median vs error quadtratic. i think always this will result in 2 less triangles. if one of the vertices is a border vertex then not collapse to middle but position of border vertex. decide which edges to collapse by the distance between the edges? smallest edge distances first? repeat this process until there are <= 128 triangles left.

    //delete all edges of the collapsed and changed triangles
    // (from triangles and map) that included a bad vertices
    //delete the two collapsed triangles from the triangles,
    //update the vertex in all triangles, if you had one of the bad update to new
    //create new edges of the updated triangles and insert into map and triangles

    from nanite: group those with the most shared edges and split if not possible to simplify to 128 into multpiple clusters

    //problem with current is that locked edges (detailed) will accumilate if you just merge and simplify linearly

    //solution: take level of clusters of entire mesh, group nearest clusters based on their bounds, then simplify each group to half the triangles and split again into clusters, as a result forming the next level of clusters (edges are only locked over 2 levels so cannot accomilate)

    //difficulty later findingcut of the dag

    //CORE:
    // X get working with all models -> solution: allowMikroCracks flag to allow collapse of smallest border edges
    // X flickering problem
    // X make instance compute, turn cluster compute into persistant thread queue,
    // X first push all via instance compute,
    // X later push only top of acceleration tree
    // X implement acceleraltion tree
    // X multiple entities / instances
    //finish geometry pipeline
    //in the end make virtual geometry exportable and loadable

    //OPTIMIZE:
    // X error random - bad? yes random is meaning less which ones in cut is random but we want in cut based on size, so error from area of cluster!!!
    // X evaluate each cluster currently
    // X -> should do tree based not evaluate children with persistant threads and atomic queue
    // X -> tree based on parenterror? for deciding to traverse children?
    // X clean up, improve code, compact everything, refactor reduce memory and redudant stuff
    // X memory leak because of keeping unused stuff?
    //threshold with object radius
    //better persistant threads: (aka use more than 32 threads) better global queue or one queue per object with its workgroup? something!! make real queue not just list like now, maybe: https://gist.github.com/Shimmen/16aabbc19feb70a4d7b9399e508d20ab
    //instance frustum culling
    //cluster frustum culling

    // X freeze mode for debug
    // X detect if queue size is too small -> visible by holes
    // X per instance shading mode
    //debug info stats
