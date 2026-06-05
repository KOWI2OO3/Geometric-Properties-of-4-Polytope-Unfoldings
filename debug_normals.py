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

print("Analyzing facet normals in detail:")
polytope_centroid = np.mean(hull.points, axis=0)
print(f"Polytope centroid: {polytope_centroid}")
print()

for i in range(len(hull.simplices)):
    simplex = hull.simplices[i]
    verts = hull.points[simplex]
    facet_centroid = np.mean(verts, axis=0)
    
    # Compute vectors from first vertex
    U = np.array([
        verts[1] - verts[0],
        verts[2] - verts[0],
        verts[3] - verts[0]
    ])
    
    # SVD
    _, S, vh = np.linalg.svd(U)
    n = vh[-1]
    n_normalized = n / np.linalg.norm(n)
    
    # Check orientation
    to_facet = facet_centroid - polytope_centroid
    dot_product = np.dot(n_normalized, to_facet)
    
    print(f"Facet {i} (vertices {simplex}):")
    print(f"  Facet centroid: {facet_centroid}")
    print(f"  To_facet vector: {to_facet}")
    print(f"  SVD components: v1={U[0]}, v2={U[1]}, v3={U[2]}")
    print(f"  Singular values: {S}")
    print(f"  SVD normal (raw): {n}")
    print(f"  SVD normal (normalized): {n_normalized}")
    print(f"  Dot product with to_facet: {dot_product}")
    print(f"  Norm of to_facet: {np.linalg.norm(to_facet)}")
    
    # Compute via Unfolder
    n_unfolder = Unfolder.facet_normal(hull, i)
    print(f"  Unfolder normal: {n_unfolder}")
    print(f"  Norm: {np.linalg.norm(n_unfolder)}")
    print()
