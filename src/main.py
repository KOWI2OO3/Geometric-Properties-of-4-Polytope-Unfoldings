# import math

# import generation as gen
# import matplotlib.pyplot as plt
# import numpy as np
# import networkx as nx
# import net
# from net import Net
# from scipy.spatial import ConvexHull
# from geometry import Facet
# import visualize as vis

# hull = gen.generate_uniform_unit_convex(30, 2)
# hull = gen.generate_clustering_unit_convex(30, 0.8, 2)
# hull = gen.generate_convext_with_deformation(30, np.array([1, 1]), 2)

# print(len(hull.points), hull.ndim)

# plt.plot(hull.points[:,0], hull.points[:,1], 'o')
# for simplex in hull.simplices:
#     plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-')

# plt.plot(hull.points[hull.vertices,0], hull.points[hull.vertices,1], 'r--', lw=2)
# plt.plot(hull.points[hull.vertices[0],0], hull.points[hull.vertices[0],1], 'ro')
# plt.show()

# Used to visualize the generation method in 2D
# while(True):
#     num_points = int((np.random.default_rng().random() + 0.05) * 100)
#     clustering = np.random.default_rng().random()
#     hull = gen.generate_clustering_unit_convex(num_points, clustering, 2)

#     print(len(hull.points), hull.ndim)

#     plt.figure()
#     plt.plot(hull.points[:,0], hull.points[:,1], 'o')
#     for simplex in hull.simplices:
#         plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-')

#     plt.plot(hull.points[hull.vertices,0], hull.points[hull.vertices,1], 'r--', lw=2)
#     plt.plot(hull.points[hull.vertices[0],0], hull.points[hull.vertices[0],1], 'ro')
#     plt.show()

# # Generation speed test
# for _ in range(0,4000):
#     num_points = int((np.random.default_rng().random() + 0.05) * 100)
#     clustering = np.random.default_rng().random()
#     hull = gen.generate_clustering_unit_convex(num_points, clustering, 2)
#     # print(len(hull.points), hull.ndim)

# Used to test facet detection
# while(True):
#     num_points = int((np.random.default_rng().random() + 0.08) * 100)
#     clustering = np.random.default_rng().random()
#     hull = gen.generate_clustering_unit_convex(num_points, clustering, 4)

#     simplex = hull.simplices[0]
#     assert(len(simplex) == 4)   # Checking tetrahydron as 3D Cell
    
#     print(type(simplex), type(simplex[0]))
#     print(len(hull.points), hull.ndim)

#     plt.figure()

#     plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'o')
    
#     plt.show()

# hull = gen.generate_clustering_unit_convex(5, 0, 4)
# G = nx.Graph()
# for i in range(0, len(hull.simplices)):
#     G.add_node(i);

# for (i, neighbors) in enumerate(hull.neighbors):
#     for j in neighbors:
#         if j != -1:
#             G.add_edge(i, int(j))

# net = nx.random_spanning_tree(G, weight=None)
# print(net.edges)

# vis.visualizeNet(net, G)

# 5 points in 4D
# This creates 2 tetrahedral facets sharing a ridge
# points = np.array([
#     [0, 0, 0, 0],  # 0
#     [1, 0, 0, 0],  # 1
#     [0, 1, 0, 0],  # 2
#     [0, 0, 1, 0],  # 3 -> facet A
#     [0, 0, 0, 1],  # 4 -> facet B
# ], dtype=float)

# hull = net.ConvexHull(points)
# graph = net.constructAdjacencyGraph(hull)
# net = net.Net.sampleRandomSpanningTree(hull, graph)
# facets = net.get_unfolded_facets()

# for i, facet in enumerate(facets):
#     print(f"\nFacet {i}")
#     print(facet.points)
#     print("w coords:", facet.points[:, 3])


# hull = gen.generate_clustering_unit_convex(40, 0)
# print(len(hull.simplices), len(hull.simplices) == 5) # should be 5

# graph, valid_indices = net.constructAdjacencyGraph(hull)
# print(graph.number_of_edges(), graph.number_of_edges() == 10)

# net = Net.sampleRandomSpanningTree(hull, graph, valid_indices)
# print(net.net.number_of_edges(), net.net.number_of_edges() == 4)
# print(net.net.number_of_nodes(), net.net.number_of_nodes() == 5)

# # for i, facet in enumerate(net.get_unfolded_facets()):
# #     print(f"Facet {i}", facet.mesh.vertices)

# print("Is overlap free: ", net.is_unfolding_overlap_free())

import time

import generation as gen
import numpy as np
from net import Net, constructAdjacencyGraph

n_samples = 10
m_instances = 5

avg_time_polytope_construction = 0
avg_time_adjacency_graph = 0
avg_time_sample_spanning_tree = 0
avg_time_intersection_test = 0
avg_facet_count = 0

intermediate_logging = False

def pipeline_sample(hull):
    start = time.time()
    graph, valid_indices = constructAdjacencyGraph(hull)
    end = time.time()
    global avg_time_adjacency_graph
    avg_time_adjacency_graph += end - start

    if intermediate_logging:
        print(f'Constructing adjacency graph took: {end - start:.2f} seconds')

    n_success: int = 0
    for i in range(n_samples):
        start = time.time()
        net = Net.sampleRandomSpanningTree(hull, graph, valid_indices)
        end = time.time()
        global avg_time_sample_spanning_tree
        avg_time_sample_spanning_tree += end - start
        if intermediate_logging:
            print(f'Sampling spanning tree took: {end - start:.2f} seconds')

        
        start = time.time()
        if net.is_unfolding_overlap_free():
            n_success += 1
        end = time.time()
        global avg_time_intersection_test
        avg_time_intersection_test += end - start
        if intermediate_logging:
            print(f'Unfolding overlap test took: {end - start:.2f} seconds')

    return float(n_success) / float(n_samples)


for m in range(m_instances):
    print("checking polytope", m)
    start = time.time()
    hull = gen.generate_clustering_unit_convex(12, 1)
    end = time.time()

    avg_time_polytope_construction += end - start
    if intermediate_logging:
        print(f'Generating polytope took: {end - start:.2f} seconds')
        print("facets:", len(hull.simplices))
    avg_facet_count += len(hull.simplices)

    success_rate = pipeline_sample(hull)

avg_facet_count /= m_instances
avg_time_polytope_construction /= m_instances
avg_time_adjacency_graph /= m_instances
avg_time_sample_spanning_tree /= m_instances * n_samples
avg_time_intersection_test /= m_instances * n_samples

print("average facet count:", avg_facet_count)
print("average time for polytope construction:", int(avg_time_polytope_construction*1000), "ms")
print("average time for adjacency graph:", int(avg_time_adjacency_graph*1000), "ms")
print("average time for sampling spanning tree:", int(avg_time_sample_spanning_tree*1000), "ms")
print("average time for testing intersection:", int(avg_time_intersection_test*1000), "ms")