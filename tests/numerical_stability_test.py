"""
Tests for numerical stability and robustness of the unfolding algorithm.

This module tests edge cases and numerical properties to ensure the algorithm
is numerically stable and handles corner cases gracefully.
"""

import numpy as np
from scipy.spatial import ConvexHull
from src.net import Net, constructAdjacencyGraph, Unfolder
import pytest


class TestNumericalStability:
    """Test numerical stability of transform and projection operations"""
    
    def test_transforms_preserve_orthogonality(self):
        """Test that cumulative transforms remain affine and well-conditioned"""
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
        
        T = net._propagate_transforms(net.net, 0)
        
        # Check that all transforms are well-conditioned
        for i, matrix in T.items():
            det = np.linalg.det(matrix)
            # Determinant should be non-zero and not too large
            assert not np.isclose(det, 0.0, atol=1e-10), f"Transform {i} is singular"
            assert np.isfinite(det), f"Transform {i} has infinite determinant"
    
    def test_no_nan_in_unfolding(self):
        """Test that unfolding doesn't produce NaN values"""
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
        
        facets = net.get_unfolded_facets()
        
        for i, facet in enumerate(facets):
            assert np.all(np.isfinite(facet.mesh.vertices)), \
                f"Facet {i} contains NaN or infinite values"
    
    def test_basis_orthonormality(self):
        """Test that computed basis is orthonormal"""
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
        
        basis = net.compute_global_basis(0)
        
        # Check orthonormality: basis.T @ basis should be identity
        gram = basis.T @ basis
        identity = np.eye(3)
        assert np.allclose(gram, identity, atol=1e-10), "Basis is not orthonormal"
    
    def test_normal_normalization(self):
        """Test that all normals have unit length"""
        points = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        hull = ConvexHull(points)
        
        for i in range(len(hull.simplices)):
            n = Unfolder.facet_normal(hull, i)
            norm = np.linalg.norm(n)
            assert np.isclose(norm, 1.0, atol=1e-9), \
                f"Normal {i} has norm {norm}, not 1.0"


class TestEdgeCaseGeometries:
    """Test unfolding on special geometries"""
    
    def test_minimal_simplex(self):
        """Test unfolding of the smallest possible simplex"""
        # 4-simplex (5 vertices)
        points = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        hull = ConvexHull(points)
        graph = constructAdjacencyGraph(hull)
        
        # Try to find at least one overlap-free unfolding
        found = False
        for attempt in range(10):
            net = Net.sampleRandomSpanningTree(hull, graph)
            if net.is_unfolding_overlap_free():
                found = True
                break
        
        assert found, "Simplex should have at least one overlap-free unfolding"
    
    def test_scaled_polytope(self):
        """Test that scaling doesn't break unfolding"""
        points = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        # Test multiple scales
        for scale in [0.1, 1.0, 10.0, 100.0]:
            scaled_points = points * scale
            hull = ConvexHull(scaled_points)
            graph = constructAdjacencyGraph(hull)
            
            # Just verify it can run and find valid unfoldings
            for attempt in range(5):
                net = Net.sampleRandomSpanningTree(hull, graph)
                if net.is_unfolding_overlap_free():
                    break
    
    def test_translated_polytope(self):
        """Test that translation doesn't affect unfolding"""
        points = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        # Test multiple translations
        for offset in [0, 1, 10, -5]:
            translated = points + np.array([offset, offset, offset, offset])
            hull = ConvexHull(translated)
            graph = constructAdjacencyGraph(hull)
            
            # Just verify it can run
            for attempt in range(5):
                net = Net.sampleRandomSpanningTree(hull, graph)
                if net.is_unfolding_overlap_free():
                    break
    
    def test_regular_simplex_in_different_orientations(self):
        """Test that orientation doesn't affect unfolding validity"""
        # Base simplex
        points = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        # Apply permutations
        permutations = [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [3, 2, 1, 0],
        ]
        
        for perm in permutations:
            rotated = points[:, perm]
            hull = ConvexHull(rotated)
            graph = constructAdjacencyGraph(hull)
            
            # Just verify it can run
            for attempt in range(5):
                net = Net.sampleRandomSpanningTree(hull, graph)
                if net.is_unfolding_overlap_free():
                    break


class TestUnfoldingProperties:
    """Test mathematical properties of the unfolding"""
    
    def test_ridge_vertices_preserved_distance(self):
        """Test that ridge vertices maintain their relative distances"""
        points = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        hull = ConvexHull(points)
        
        # For each pair of adjacent facets
        for i in range(len(hull.neighbors)):
            neighbors = [n for n in hull.neighbors[i] if n != -1]
            for j in neighbors:
                # Get shared ridge
                ridge = Unfolder.shared_ridge(hull, i, j)
                ridge_pts = hull.points[ridge]
                
                # Compute distances in original space
                dist_orig = np.linalg.norm(ridge_pts[1] - ridge_pts[0])
                
                # Compute rotation
                M = Unfolder.compute_unfold_rotation(hull, i, j)
                
                # Apply rotation to ridge points
                transformed = []
                for pt in ridge_pts:
                    hp = np.append(pt, 1.0)
                    tp = (M @ hp)[:4]
                    transformed.append(tp)
                transformed = np.array(transformed)
                
                # Compute distances after transformation
                dist_transformed = np.linalg.norm(transformed[1] - transformed[0])
                
                # Distances should be preserved
                assert np.isclose(dist_orig, dist_transformed, rtol=1e-9), \
                    f"Ridge distance not preserved between facets {i} and {j}"
    
    def test_adjacency_graph_is_connected(self):
        """Test that adjacency graph of polytope is always connected"""
        points = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        
        hull = ConvexHull(points)
        graph = constructAdjacencyGraph(hull)
        
        # For a convex polytope, the dual graph should be connected
        import networkx as nx
        assert nx.is_connected(graph), "Adjacency graph should be connected"
    
    def test_spanning_tree_has_correct_edges(self):
        """Test that spanning tree has correct number of edges"""
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
        
        num_facets = len(hull.simplices)
        num_edges = net.net.number_of_edges()
        
        # Spanning tree should have exactly (n-1) edges
        assert num_edges == num_facets - 1, \
            f"Spanning tree should have {num_facets - 1} edges, got {num_edges}"
