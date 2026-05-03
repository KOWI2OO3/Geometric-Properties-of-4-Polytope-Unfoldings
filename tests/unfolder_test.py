from src.net import Unfolder
import numpy as np

def test_shared_ridge(simplex_hull):
    ridge = Unfolder.shared_ridge(simplex_hull, 0, 1)
    assert len(ridge) == 3

def test_unfold_rotation_preserves_ridge(simplex_hull):
    M = Unfolder.compute_unfold_rotation(simplex_hull, 0, 1)

    ridge = Unfolder.shared_ridge(simplex_hull, 0, 1)

    for idx in ridge:
        p = simplex_hull.points[idx]
        hp = np.append(p, 1.0)

        transformed = (M @ hp)[:4]

        assert np.allclose(
            transformed,
            p,
            atol=1e-8
        )

def test_unfold_aligns_normals(simplex_hull):
    M = Unfolder.compute_unfold_rotation(simplex_hull, 0, 1)

    facet = simplex_hull.simplices[1]
    points = simplex_hull.points[facet]

    transformed = []

    for p in points:
        hp = np.append(p, 1.0)
        transformed.append((M @ hp)[:4])

    transformed = np.array(transformed)

    # recompute normal manually
    U = np.array([
        transformed[1] - transformed[0],
        transformed[2] - transformed[0],
        transformed[3] - transformed[0]
    ])

    _, _, vh = np.linalg.svd(U)
    new_n = vh[-1]
    new_n /= np.linalg.norm(new_n)

    old_n = Unfolder.facet_normal(simplex_hull, 0)

    assert abs(np.dot(new_n, old_n)) > 0.999

def test_rotation_preserves_ridge(simplex_hull):
    M = Unfolder.compute_unfold_rotation(simplex_hull, 0, 1)
    ridge = Unfolder.shared_ridge(simplex_hull, 0, 1)

    for i in ridge:
        p = simplex_hull.points[i]
        hp = np.append(p, 1.0)

        tp = (M @ hp)[:4]

        assert np.allclose(tp, p, atol=1e-8)

def test_unfolding_runs(net):
    facets = net.get_unfolded_facets()
    assert len(facets) == 5

def test_facets_have_valid_geometry(net):
    facets = net.get_unfolded_facets()

    for f in facets:
        assert f.mesh.vertices.shape[1] == 3
        assert len(f.mesh.vertices) >= 3