from __future__ import annotations
from scipy.spatial import ConvexHull
import networkx as nx
import numpy as np
from geometry import Facet

class Net:
    hull: ConvexHull
    net: nx.Graph

    # Maps node index -> original hull.simplices index.
    _valid_indices: list[int]

    def __init__(self, hull: ConvexHull, net: nx.Graph, valid_indices: list[int]):
        self.hull = hull
        self.net = net
        self._valid_indices = valid_indices

    @staticmethod
    def sampleRandomSpanningTree(hull: ConvexHull, graph: nx.Graph, valid_indices: list[int]) -> Net:
        net = nx.random_spanning_tree(graph, weight=None)
        return Net(hull, net, valid_indices)

    def is_unfolding_overlap_free(self) -> bool:
        """
        True if no two non-adjacent unfolded facets interpenetrate.

        Adjacent facets (connected in the spanning tree) share a ridge and are
        allowed to touch along it.  All other pairs must not intersect.
        """
        facets = self.get_unfolded_facets()
        n = len(facets)
        for i in range(n):
            for j in range(i + 1, n):
                if self.net.has_edge(i, j):
                    continue
                if facets[i].intersects(facets[j]):
                    return False
        return True

    def check_unfolding_validity(self) -> dict:
        facets = self.get_unfolded_facets()
        n = len(facets)
        collisions = []
        for i in range(n):
            for j in range(i + 1, n):
                if self.net.has_edge(i, j):
                    continue
                if facets[i].intersects(facets[j]):
                    collisions.append((i, j))
        return {
            'num_facets': n,
            'non_adjacent_collisions': collisions,
            'is_overlap_free': len(collisions) == 0,
        }

    def get_unfolded_facets(self) -> list[Facet]:
        """
        Unfold all (non-degenerate) facets into 3D and return them as Facet objects.

        Pipeline
        --------
        1.  Choose a root facet and build Q: the (4×3) orthonormal basis of its
            3-D hyperplane in R^4 (via QR decomposition of its edge vectors).
        2.  DFS over the spanning tree, accumulating a 5×5 affine transform T[i]
            for every node i.  T[root] = I; T[child] = T[parent] @ local_unfold.
            local_unfold rotates child flat into parent's current hyperplane.
        3.  For each node i, apply T[i] to its four 4-D vertices (in homogeneous
            form) and then project the result onto Q to obtain 3-D coordinates.

        Why the composition order T[parent] @ local_unfold is correct
        --------------------------------------------------------------
        We work with column-convention homogeneous coordinates:
            p_new = M @ p_old
        local_unfold rotates child relative to parent *in the original 4-D frame*.
        T[parent] then maps that already-unfolded result into the root's hyperplane.
        So the combined action is: first unfold child, then carry it along with
        the parent — which is exactly what we want.

        Why Q gives a proper tetrahedron
        ---------------------------------
        Q spans the root facet's hyperplane.  After T[i] is applied, facet i lies
        in that same hyperplane, so x4 -> x4 @ Q  (equivalently Q.T @ x4) is a
        bijection onto R^3, and a solid tetrahedron maps to a solid tetrahedron.
        """
        root = next(iter(self.net.nodes()))
        basis = self.compute_global_basis(root)          # (4, 3)
        T = self._propagate_transforms(self.net, root)   # node → 5×5

        facets = []
        for node in sorted(self.net.nodes()):
            simplex_idx = self._valid_indices[node]
            points = self.hull.points[self.hull.simplices[simplex_idx]]  # (4, 4)

            points_3d = []
            for point in points:
                ph = np.append(point, 1.0)     # homogeneous 5-vector
                tp = T[node] @ ph              # apply cumulative affine transform
                point_4d = tp[:4]                   # drop homogeneous coordinate
                points_3d.append(point_4d @ basis)      # project 4-D → 3-D

            facets.append(Facet(np.array(points_3d)))

        return facets

    def compute_global_basis(self, root_node: int) -> np.ndarray:
        """
        (4, 3) orthonormal basis of the root facet's hyperplane in R^4.

        Built by QR-decomposing the 4×3 matrix whose columns are the three
        edge vectors of the root tetrahedron.
        """
        simplex_idx = self._valid_indices[root_node]
        points = self.hull.points[self.hull.simplices[simplex_idx]]   # (4, 4)
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        v3 = points[3] - points[0]

        M = np.stack([v1, v2, v3], axis = 1) # 4x3 matrix

        # orthonormalize using QR decomposition
        Q, _ = np.linalg.qr(M)   # Q is 4x3

        return Q

    def _propagate_transforms(self, tree: nx.Graph, root: int) -> dict:
        """
        DFS over the spanning tree to accumulate 5×5 affine transforms.

        T[root] = I.
        T[child] = T[parent] @ local_unfold_rotation(parent → child).

        The local rotation unfolds child relative to parent by rotating around
        their shared ridge until the two facets are coplanar in the same
        hyperplane (180° "dihedral" in 4-D terms).
        """
        T = { root: np.eye(5) }

        def dfs(parent: int):
            for child in tree.neighbors(parent):
                # Avoid cycles by checking if already visited
                if child in T:
                    continue

                local_M = Unfolder.compute_unfold_rotation(
                    self.hull,
                    self._valid_indices[parent],
                    self._valid_indices[child],
                )
                T[child] = T[parent] @ local_M
                dfs(child)

        dfs(root)
        return T

class Unfolder:
    
    @staticmethod
    def compute_unfold_rotation(hull: ConvexHull, a: int, b: int) -> np.ndarray:
        """
        5×5 affine transformation that rotates facet b flat relative to facet a.

        The key insight for 4-D unfolding
        -----------------------------------
        In 3-D (unfolding a polyhedron into 2-D), two adjacent faces that have
        been "flattened" end up lying in the same plane with their outward normals
        pointing in *opposite* directions (one up, one down from the plane).

        In 4-D (unfolding a polychoron into 3-D), two adjacent tetrahedral cells
        that have been "flattened" end up lying in the same 3-D hyperplane.  Their
        4-D outward normals both point *in the same direction* — perpendicular to
        that shared hyperplane, outward from the original polytope.

        Therefore the target direction for b's projected normal is  +a2  (same as
        a's projected normal) — NOT  -a2 as one might naively copy from the 3-D
        analogy.  Using -a2 produces a rotation of ~180° in most cases, which
        folds the facet *back* onto itself rather than flattening it.

        Algorithm
        ---------
        1.  Find the shared ridge (3 vertices → a triangle in R^4).
        2.  QR-decompose the ridge edge vectors to get a 2-D orthonormal basis
            r_basis for the ridge plane.
        3.  Take the full SVD of r_basis.T to get the 2-D orthonormal complement
            comp_basis (the 4-D directions perpendicular to the ridge).
        4.  Project both outward 4-D normals onto comp_basis to get 2-D unit
            vectors a2 and b2.
        5.  Compute the minimal signed angle θ from b2 to *+a2* using atan2.
            This is the smallest rotation needed to make both normals point the
            same way — i.e., both cells lying flat in a common hyperplane.
        6.  Embed the 2-D rotation by θ back into R^4 acting only on comp_basis.
        7.  Wrap as a 5×5 homogeneous rotation around the ridge's pivot vertex.

        Args:
            hull : ConvexHull of the 4-polytope.
            a    : simplex index of the parent (reference) facet.
            b    : simplex index of the child facet (must be adjacent to a).

        Returns:
            np.ndarray: (5, 5) affine transformation matrix.
        """
        points = hull.points

        # Constructing ridge
        ridge = Unfolder.shared_ridge(hull, a, b)
        ridge_pts = points[ridge] # (3, 4)
        pivot = ridge_pts[0].copy()

        # Computing ridge basis (2-D plane in R^4 spanned by the ridge)
        r1 = ridge_pts[1] - ridge_pts[0]
        r2 = ridge_pts[2] - ridge_pts[0]
        ridge_basis, _ = np.linalg.qr(np.stack([r1, r2], axis=1))  # (4, 2)

        # Compute orthogonal complement (2-D plane is perpendicular to ridge in R^4)
        # The last two right-singular vectors of ridge_basis.T span the complement.
        _, _, vh = np.linalg.svd(ridge_basis.T, full_matrices=True)
        comp_basis = vh[2:].T # (4, 2)

        # Project normals outward onto comp_basis
        nA = Unfolder.facet_normal(hull, a)
        nB = Unfolder.facet_normal(hull, b)

        a2 = comp_basis.T @ nA;  
        a2 /= np.linalg.norm(a2)
        b2 = comp_basis.T @ nB;  
        b2 /= np.linalg.norm(b2)

        # Compute minimal signed angle from b2 to +a2
        # TARGET IS +a2, NOT -a2.
        target  = a2
        cross_2d = b2[0] * target[1] - b2[1] * target[0]
        dot_2d   = b2[0] * target[0]  + b2[1] * target[1]
        theta   = np.arctan2(cross_2d, dot_2d)

        # Lift 2-D rotation into R^4 via comp_basis
        c = np.cos(theta)
        s = np.sin(theta)
        rotation_2d = np.array([[c, -s], [s, c]])
        R    = np.eye(4) + comp_basis @ (rotation_2d - np.eye(2)) @ comp_basis.T

        # Build affine rotation around pivot
        return Unfolder.build_affine_rotation(R, pivot)

    @staticmethod
    def shared_ridge(hull: ConvexHull, a: int, b: int) -> np.ndarray:
        """
        3 shared vertex indices between two adjacent 4-D tetrahedral facets.

        Raises ValueError if the intersection is not exactly 3 vertices.
        """
        shared = sorted(set(hull.simplices[a]) & set(hull.simplices[b]))
        if len(shared) != 3:
            raise ValueError(
                f"Expected 3 shared vertices (triangular ridge) between "
                f"facets {a} and {b}, got {len(shared)}: {shared}"
            )
        return np.array(shared, dtype=int)

    @staticmethod
    def facet_normal(hull: ConvexHull, facet: int) -> np.ndarray:
        """
        Outward-pointing unit normal of a tetrahedral facet of the 4-polytope.

        The normal is the null-space vector of the (3×4) edge matrix, oriented
        away from the polytope centroid.
        """
        verts = hull.points[hull.simplices[facet]]   # (4, 4)
        edges = np.array([
            verts[1] - verts[0],
            verts[2] - verts[0],
            verts[3] - verts[0]
        ])                                           # (3, 4)

        _, _, vh = np.linalg.svd(edges, full_matrices=True)
        n = vh[-1]
        n /= np.linalg.norm(n)

        facet_centroid    = verts.mean(axis=0)
        polytope_centroid = hull.points.mean(axis=0)
        if np.dot(n, facet_centroid - polytope_centroid) < 0:
            n = -n

        return n

    @staticmethod
    def build_affine_rotation(R: np.ndarray, pivot: np.ndarray) -> np.ndarray:
        """
        5×5 homogeneous matrix encoding the rotation  p → R @ (p − pivot) + pivot.
        """
        M = np.eye(5)
        M[:4, :4] = R
        M[:4, 4]  = pivot - R @ pivot
        return M

def constructAdjacencyGraph(hull: ConvexHull) -> tuple[nx.Graph, list[int]]:
    """
    Dual graph of the polytope with degenerate simplex filtering.

    WHY we filter
    -------------
    scipy's ConvexHull triangulates the boundary into simplices.  For polytopes
    whose facets are not simplices (e.g. the hypercube), some sub-simplices are
    degenerate: their 4 vertices are coplanar in 4-D (rank < 3).  Including them
    breaks unfolding because their 4-D normal is ill-defined and their 3-D
    projection has zero volume, causing spurious trimesh collisions.

    We keep only simplices whose edge matrix has rank 3.

    Returns
    -------
    graph        : nx.Graph with nodes 0..K-1 (K = number of valid simplices),
                   edges between adjacent valid simplices.
    valid_indices: list mapping node id → original hull.simplices index.
    """
    n_all = len(hull.simplices)

    valid_indices = []
    for i in range(n_all):
        pts = hull.points[hull.simplices[i]]
        edges = pts[1:] - pts[0]       # (3, 4)
        if np.linalg.matrix_rank(edges) == 3:
            valid_indices.append(i)

    valid_set = set(valid_indices)
    original_to_node = {}
    for node, original in enumerate(valid_indices):
        original_to_node[original] = node

    graph = nx.Graph()
    for node in range(len(valid_indices)):
        graph.add_node(node)

    for node, orig_i in enumerate(valid_indices):
        for orig_j in hull.neighbors[orig_i]:
            if orig_j != -1 and orig_j in valid_set:
                graph.add_edge(node, original_to_node[orig_j])

    return graph, valid_indices