import numpy as np
from scipy.spatial import ConvexHull
from src.net import constructAdjacencyGraph

def test_4_simplex_graph():
    points = np.array([
        [0,0,0,0],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ], dtype=float)

    hull = ConvexHull(points)

    graph = constructAdjacencyGraph(hull)

    assert len(hull.simplices) == 5
    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 10