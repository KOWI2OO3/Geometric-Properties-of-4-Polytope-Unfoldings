from __future__ import annotations
from scipy.spatial import ConvexHull
import networkx as nx
import numpy as np
from geometry import Facet

class Net:
    hull: ConvexHull
    net: nx.Graph

    def __init__(self, hull: ConvexHull, net: nx.Graph):
        self.hull = hull
        self.net = net

    @staticmethod
    def sampleRandomSpanningTree(hull: ConvexHull, graph: nx.Graph) -> Net:
        net = nx.random_spanning_tree(graph, weight=None)
        return Net(hull, net)
    
    def is_unfolding_overlap_free(self) -> bool:
        facets = self.get_unfolded_facets()

        state = True
        for i, a in enumerate(facets):
            for j, b in enumerate(facets):
                if i >= j or self.net.has_edge(i, j):
                    continue
                
                if a.intersects(b):
                    return False
        return True



    def get_unfolded_facets(self) -> list[Facet]:
        basis = self.compute_global_basis(0)
        
        facets = []
        T = self._propagate_transforms(self.net, 0)
        for i, simplex in enumerate(self.hull.simplices):
            points = self.hull.points[simplex]
            facet_points = []
            for point in points:
                p = np.append(point, 1.0)
                tp = T[i] @ p
                tp4 = tp[:4]
                facet_points.append(self.project_to_3d(tp4, basis))
            facets.append(Facet(np.array(facet_points)))
        return facets
    
    def project_to_3d(self, x4: np.ndarray, basis: np.ndarray) -> np.ndarray:
        return x4 @ basis

    def compute_global_basis(self, root_facet: int) -> np.ndarray:
        points = np.array(self.hull.points[self.hull.simplices[root_facet]])

        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        v3 = points[3] - points[0]

        M = np.stack([v1, v2, v3], axis = 1) # 4x3 matrix

        # orthonormalize
        Q, _ = np.linalg.qr(M)   # Q is 4x3

        return Q

    def _propagate_transforms(self, tree: nx.Graph, root: int):
        T = { root: np.eye(5) }

        def dfs(parent):
            for child in tree.neighbors(parent):
                # Ensure we don't run into cycles and we do not go back in the graph as the graph is not directed
                if child in T:
                    continue
                    
                local_M = Unfolder.compute_unfold_rotation(self.hull, parent, child)
                T[child] = T[parent] @ local_M
                dfs(child)
        
        dfs(root)
        return T

class Unfolder:
    
    @staticmethod
    def compute_unfold_rotation(hull: ConvexHull, a: int, b: int) -> np.ndarray:
        points = hull.points

        # Shared ridge
        ridge = Unfolder.shared_ridge(hull, a, b)
        ridge_pts = points[ridge]

        pivot = ridge_pts[0]

        # Ridge basis (2D plane in R4)
        r1 = ridge_pts[1] - ridge_pts[0]
        r2 = ridge_pts[2] - ridge_pts[0]

        ridge_basis, _ = np.linalg.qr(
            np.stack([r1, r2], axis=1)
        )  # 4x2

        # Orthogonal complement (2D)
        _, _, vh = np.linalg.svd(ridge_basis.T)
        comp_basis = vh[2:].T  # 4x2

        # Facet normals
        nA = Unfolder.facet_normal(hull, a)
        nB = Unfolder.facet_normal(hull, b)

        # Coordinates inside complement
        a2 = comp_basis.T @ nA
        b2 = comp_basis.T @ nB

        a2 /= np.linalg.norm(a2)
        b2 /= np.linalg.norm(b2)

        # 2D angle
        angleA = np.arctan2(a2[1], a2[0])
        angleB = np.arctan2(b2[1], b2[0])

        # Compute the rotation angle
        # We rotate facet b so its normal points opposite to facet a's normal
        # within the orthogonal complement of the shared ridge.
        theta = angleA - angleB + np.pi

        c = np.cos(theta)
        s = np.sin(theta)

        rot2 = np.array([
            [c, -s],
            [s,  c]
        ])

        # Lift back into R4
        R = np.eye(4)
        R += comp_basis @ (rot2 - np.eye(2)) @ comp_basis.T

        return Unfolder.build_affine_rotation(R, pivot)

    @staticmethod
    def shared_ridge(hull: ConvexHull, a: int, b: int) -> np.ndarray: 
        fa = hull.simplices[a]
        fb = hull.simplices[b]

        shared = list(set(fa) & set(fb))
        if(len(shared) != 3):
            raise ValueError("Expected triangular ridge in 4D facet adjacency")
        
        return np.array(shared, dtype=int)

    @staticmethod
    def facet_normal(hull : ConvexHull, facet: int) -> np.ndarray:
        verts = hull.points[hull.simplices[facet]]
        U = np.array([
            verts[1] - verts[0],
            verts[2] - verts[0],
            verts[3] - verts[0]
        ])

        _, _, vh = np.linalg.svd(U)
        n = vh[-1]
        n = n / np.linalg.norm(n)
        
        # Ensure normal points outward from polytope centroid
        facet_centroid = np.mean(verts, axis=0)
        polytope_centroid = np.mean(hull.points, axis=0)
        to_facet = facet_centroid - polytope_centroid
        
        if np.dot(n, to_facet) < 0:
            n = -n
        
        return n

    @staticmethod
    def align_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
        v_from /= np.linalg.norm(v_from)
        v_to /= np.linalg.norm(v_to)

        v = v_from + v_to
        if np.linalg.norm(v) < 1e-12:
            # opposite vectors -> 180 degree rotation
            v = np.zeros_like(v_from)
            v[0] = 1.0
        
        v /= np.linalg.norm(v)
        return 2 * np.outer(v, v) - np.eye(len(v_from))

    @staticmethod
    def build_affine_rotation(R: np.ndarray, pivot: np.ndarray) -> np.ndarray:
        M = np.eye(5)
        M[:4, :4] = R
        M[:4, 4] = pivot - R @ pivot
        return M    

def constructAdjacencyGraph(hull: ConvexHull) -> nx.Graph:
    graph = nx.Graph()
    for i in range(0, len(hull.simplices)):
        graph.add_node(i);

    for (i, neighbors) in enumerate(hull.neighbors):
        for j in neighbors:
            if j != -1:
                graph.add_edge(i, int(j))
    return graph
