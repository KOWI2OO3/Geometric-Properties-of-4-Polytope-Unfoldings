import numpy as np
from src.geometry import Facet

def test_separated():
    a = Facet(np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ], dtype=float))

    b = Facet(np.array([
        [5,5,5],
        [6,5,5],
        [5,6,5],
        [5,5,6]
    ], dtype=float))

    assert not a.intersects(b)


def test_overlapping():
    a = Facet(np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ], dtype=float))

    b = Facet(np.array([
        [0.2,0.2,0.2],
        [1.2,0.2,0.2],
        [0.2,1.2,0.2],
        [0.2,0.2,1.2]
    ], dtype=float))

    assert a.intersects(b)


def test_touching():
    a = Facet(np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ], dtype=float))

    b = Facet(np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,-1]
    ], dtype=float))

    assert not a.intersects(b)

def test_no_spurious_self_collisions(net):
    facets = net.get_unfolded_facets()

    for i, a in enumerate(facets):
        for j, b in enumerate(facets):
            if i == j:
                continue

            # adjacency allowed
            if net.net.has_edge(i, j):
                continue

            assert not a.intersects(b), f"Collision: {i} {j}"

def test_centroid_stability(net):
    facets = net.get_unfolded_facets()

    centroids = [f.mesh.vertices.mean(axis=0) for f in facets]

    # no NaNs / infinities
    for c in centroids:
        assert np.isfinite(c).all()