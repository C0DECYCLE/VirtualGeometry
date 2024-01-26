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

1. import and parse geometry
2. divde into clusters, merge upwards and simplify and build tree
   (binary, quad or dag) encode in linear format
3. build attributes for all clusters
   for each instance (=root cluster) build its info

### Runtime

0. (first sync cpu-gpu object changes?)
1. for each instance cull in compute (reduce as much as possible) (go trough instance buffer and write survived ids into new buffer)
2. for each surviving instance and its geometry tree determine the lod level and cull (reduce as much as possible)
3. add all still remaining clusters (aka its instance id and cluster id) to a (draw) buffer
4. issue one non indexed draw call on that buffer (vertexCount = 128 tris \* 3, instanceCount = number of clusters)
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
-   then do gpu culling etc, then merge and simplify and smart schedule
-   atomic also on arrays!

### TODO

-   Entity deletion
