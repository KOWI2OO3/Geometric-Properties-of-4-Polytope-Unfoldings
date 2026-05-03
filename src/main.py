import math

import generation as gen
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import net
from net import Net
from scipy.spatial import ConvexHull
from geometry import Facet
import visualize as vis

# hull = gen.generate_uniform_unit_convex(30, 2)
# hull = gen.generate_clustering_unit_convex(120, 1, 2)

# print(len(hull.points), hull.ndim)


# plt.plot(hull.points[:,0], hull.points[:,1], 'o')
# for simplex in hull.simplices:
#     plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-')

# plt.plot(hull.points[hull.vertices,0], hull.points[hull.vertices,1], 'r--', lw=2)
# plt.plot(hull.points[hull.vertices[0],0], hull.points[hull.vertices[0],1], 'ro')
# plt.show()

# # Used to visualize the generation method in 2D
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


points = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

hull = ConvexHull(points)
print(len(hull.simplices), len(hull.simplices) == 5) # should be 5

graph = net.constructAdjacencyGraph(hull)
print(graph.number_of_edges(), graph.number_of_edges() == 10)

net = Net.sampleRandomSpanningTree(hull, graph)
print(net.net.number_of_edges(), net.net.number_of_edges() == 4)
print(net.net.number_of_nodes(), net.net.number_of_nodes() == 5)

for i, facet in enumerate(net.get_unfolded_facets()):
    print(f"Facet {i}", facet.mesh.vertices)

print("Is overlap free: ", net.is_unfolding_overlap_free())