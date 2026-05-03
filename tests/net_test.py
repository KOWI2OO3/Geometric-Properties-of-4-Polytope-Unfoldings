import numpy as np
from src.net import Net, constructAdjacencyGraph

def test_simplex_unfolding_runs(simplex_hull):
    graph = constructAdjacencyGraph(simplex_hull)

    net = Net.sampleRandomSpanningTree(
        simplex_hull,
        graph
    )

    facets = net.get_unfolded_facets()

    assert len(facets) == 5    

def test_transforms_exist(net):
    T = net._propagate_transforms(net.net, 0)
    assert len(T) == net.net.number_of_nodes()

def test_root_identity(net):
    T = net._propagate_transforms(net.net, 0)
    assert np.allclose(T[0], np.eye(5))

def test_transform_is_affine(net):
    T = net._propagate_transforms(net.net, 0)

    M = T[0]
    assert M.shape == (5,5)
    assert np.isclose(M[4,4], 1.0)