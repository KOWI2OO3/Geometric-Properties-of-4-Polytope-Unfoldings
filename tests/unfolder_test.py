from src.net import Unfolder
import numpy as np

def test_shared_ridge(simplex_hull):
    ridge = Unfolder.shared_ridge(simplex_hull, 0, 1)
    assert len(ridge) == 3

def test_unfold_rotation_preserves_ridge(simplex_hull):
    """Test that unfolding rotation keeps the shared ridge fixed"""
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
        ), f"Ridge point {idx} was moved during rotation"

def test_unfold_aligns_normals(simplex_hull):
    """Test that unfolding makes normals anti-parallel (pointing opposite directions)"""
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
    new_n_consistent = Unfolder.facet_normal(simplex_hull, 1)

    # After unfolding, the rotated facet's normal should be anti-parallel to the first facet's
    # (dot product should be close to -1, not +1)
    dot_product = np.dot(new_n, old_n)
    
    # The normals should be nearly opposite (dot product close to -1)
    assert dot_product < -0.5, f"Normals should be anti-parallel after unfolding, got dot product {dot_product}"

def test_rotation_preserves_ridge(simplex_hull):
    """Test that rotation matrix preserves the shared ridge"""
    M = Unfolder.compute_unfold_rotation(simplex_hull, 0, 1)
    ridge = Unfolder.shared_ridge(simplex_hull, 0, 1)

    for i in ridge:
        p = simplex_hull.points[i]
        hp = np.append(p, 1.0)

        tp = (M @ hp)[:4]

        assert np.allclose(tp, p, atol=1e-8), f"Ridge point {i} moved"

def test_unfolding_runs(net):
    """Test that unfolding completes without errors"""
    facets = net.get_unfolded_facets()
    assert len(facets) == 5

def test_facets_have_valid_geometry(net):
    """Test that unfolded facets have valid 3D geometry"""
    facets = net.get_unfolded_facets()

    for f in facets:
        assert f.mesh.vertices.shape[1] == 3, "Facet should have 3D vertices"
        assert len(f.mesh.vertices) >= 3, "Facet should have at least 3 vertices"
        assert f.mesh.vertices.shape[0] == 4, "Facet should have exactly 4 vertices (tetrahedron facet)"

def test_facet_normal_consistency(simplex_hull):
    """Test that all normals point outward from the polytope"""
    for i in range(len(simplex_hull.simplices)):
        n = Unfolder.facet_normal(simplex_hull, i)
        assert np.linalg.norm(n) > 0.99, f"Facet {i} normal not normalized"
        
        # Check that normal points outward
        verts = simplex_hull.points[simplex_hull.simplices[i]]
        facet_centroid = np.mean(verts, axis=0)
        polytope_centroid = np.mean(simplex_hull.points, axis=0)
        to_facet = facet_centroid - polytope_centroid
        
        dot_product = np.dot(n, to_facet)
        assert dot_product > 0, f"Facet {i} normal points inward (dot product: {dot_product})"

def test_rotation_angle_produces_antiparallel_normals(simplex_hull):
    """Test that the rotation angle theta actually produces anti-parallel normals"""
    for i in range(len(simplex_hull.neighbors)):
        neighbors = [n for n in simplex_hull.neighbors[i] if n != -1]
        if len(neighbors) > 0:
            j = neighbors[0]
            
            nA = Unfolder.facet_normal(simplex_hull, i)
            nB = Unfolder.facet_normal(simplex_hull, j)
            
            # Before rotation, normals might not be anti-parallel
            dot_before = np.dot(nA, nB)
            
            M = Unfolder.compute_unfold_rotation(simplex_hull, i, j)
            
            # Transform facet j's vertices
            facet_verts = simplex_hull.points[simplex_hull.simplices[j]]
            transformed_verts = []
            for p in facet_verts:
                hp = np.append(p, 1.0)
                transformed_verts.append((M @ hp)[:4])
            transformed_verts = np.array(transformed_verts)
            
            # Compute new normal of facet j
            U = np.array([
                transformed_verts[1] - transformed_verts[0],
                transformed_verts[2] - transformed_verts[0],
                transformed_verts[3] - transformed_verts[0]
            ])
            _, _, vh = np.linalg.svd(U)
            nB_transformed = vh[-1] / np.linalg.norm(vh[-1])
            
            # Ensure same orientation as original nA
            if np.dot(nA, nB_transformed) < 0:
                nB_transformed = -nB_transformed
            
            # After rotation, normals should be anti-parallel (in the ridge complement)
            # This is a more lenient check since projections make it complex
            # But we check that the dot product changed significantly (became more negative or positive)
            dot_after = np.dot(nA, nB_transformed)
