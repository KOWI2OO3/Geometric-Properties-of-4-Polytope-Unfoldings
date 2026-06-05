import numpy as np
from scipy.spatial import ConvexHull
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.net import Net, constructAdjacencyGraph, Unfolder

# Create a 4-simplex
points = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

hull = ConvexHull(points)
graph = constructAdjacencyGraph(hull)
net = Net.sampleRandomSpanningTree(hull, graph)

print("Testing unfolding...")
print(f"Hull has {len(hull.simplices)} facets")
print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
print(f"Spanning tree has {net.net.number_of_nodes()} nodes and {net.net.number_of_edges()} edges")

# Get validity check
validity = net.check_unfolding_validity()
print(f"\nValidity check:")
print(f"  Is overlap free: {validity['is_overlap_free']}")
print(f"  Num facets: {validity['num_facets']}")
print(f"  All facets 3D: {validity['all_facets_3d']}")
print(f"  Collisions: {validity['non_adjacent_collisions']}")

# Check facets in detail
facets = net.get_unfolded_facets()
print(f"\nFacet coordinates:")
for i, facet in enumerate(facets):
    print(f"Facet {i}:")
    print(f"  Vertices shape: {facet.mesh.vertices.shape}")
    print(f"  Vertices:\n{facet.mesh.vertices}")
    print()

# Check some normals
print("Facet normals:")
for i in range(len(hull.simplices)):
    n = Unfolder.facet_normal(hull, i)
    print(f"  Facet {i}: {n}")
