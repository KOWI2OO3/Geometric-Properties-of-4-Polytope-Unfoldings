import pytest
import numpy as np
from scipy.spatial import ConvexHull
from src.net import Net, constructAdjacencyGraph

@pytest.fixture
def simplex_hull():
    points = np.array([
        [0,0,0,0],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ])
    return ConvexHull(points)

@pytest.fixture
def graph(simplex_hull):
    return constructAdjacencyGraph(simplex_hull)

@pytest.fixture
def net(simplex_hull, graph):
    return Net.sampleRandomSpanningTree(simplex_hull, graph)