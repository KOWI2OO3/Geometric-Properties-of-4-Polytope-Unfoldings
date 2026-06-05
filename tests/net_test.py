import numpy as np
from src.net import Net, constructAdjacencyGraph, Unfolder

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

def test_unfolding_is_overlap_free(simplex_hull):
    """Test that at least some unfoldings of a 4-simplex are overlap-free"""
    graph = constructAdjacencyGraph(simplex_hull)
    
    # Try multiple spanning trees - at least one should be overlap-free
    found_overlap_free = False
    for attempt in range(10):
        net = Net.sampleRandomSpanningTree(simplex_hull, graph)
        if net.is_unfolding_overlap_free():
            found_overlap_free = True
            break
    
    assert found_overlap_free, "At least one spanning tree should produce an overlap-free unfolding"

def test_unfolding_preserves_facet_count(net):
    """Test that all facets are present in the unfolding"""
    facets = net.get_unfolded_facets()
    assert len(facets) == len(net.hull.simplices), "Unfolding should preserve all facets"

def test_unfolded_facets_are_3d(net):
    """Test that all unfolded facets are 3D"""
    facets = net.get_unfolded_facets()
    for i, facet in enumerate(facets):
        assert facet.mesh.vertices.shape[1] == 3, f"Facet {i} should have 3D vertices"

def test_adjacent_facets_can_touch(simplex_hull):
    """Test that adjacent facets are allowed to touch in the unfolding"""
    graph = constructAdjacencyGraph(simplex_hull)
    net = Net.sampleRandomSpanningTree(simplex_hull, graph)
    facets = net.get_unfolded_facets()
    
    # For each edge in the net, adjacent facets should be allowed to touch/intersect
    # (this is expected and acceptable)
    for i, j in net.net.edges():
        # This doesn't assert anything, just documents that adjacent facets can intersect
        facets[i].intersects(facets[j])

def test_facet_transforms_are_consistent(simplex_hull):
    """Test that transforms are computed consistently for all facets"""
    graph = constructAdjacencyGraph(simplex_hull)
    net = Net.sampleRandomSpanningTree(simplex_hull, graph)
    T = net._propagate_transforms(net.net, 0)
    
    # All transforms should be 5x5 affine matrices
    for i in range(len(net.hull.simplices)):
        assert T[i].shape == (5, 5), f"Transform for facet {i} has wrong shape"
        assert np.isclose(T[i][4, 4], 1.0), f"Transform for facet {i} is not affine"

def test_multiple_spanning_trees_are_overlap_free(simplex_hull):
    """Test that at least some spanning trees produce overlap-free unfoldings"""
    graph = constructAdjacencyGraph(simplex_hull)
    
    # Try multiple spanning trees - expect that some are overlap-free
    overlap_free_count = 0
    for trial in range(10):
        net = Net.sampleRandomSpanningTree(simplex_hull, graph)
        if net.is_unfolding_overlap_free():
            overlap_free_count += 1
    
    assert overlap_free_count > 0, "At least some spanning trees should produce overlap-free unfoldings"

def test_unfolding_facet_no_duplicates(net):
    """Test that unfolded facets are not duplicated"""
    facets = net.get_unfolded_facets()
    
    # Compare facet coordinates to ensure they're not identical
    for i in range(len(facets)):
        for j in range(i + 1, len(facets)):
            # At least one coordinate should be different
            if not net.net.has_edge(i, j):
                # Non-adjacent facets definitely shouldn't be identical
                assert not np.allclose(facets[i].mesh.vertices, facets[j].mesh.vertices), \
                    f"Facets {i} and {j} should not be identical"
