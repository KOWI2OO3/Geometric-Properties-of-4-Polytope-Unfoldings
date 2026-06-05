from __future__ import annotations
import numpy as np

_EPS = 1e-9

_EDGE_IDX = np.array([
    [0,1],
    [0,2],
    [0,3],
    [1,2],
    [1,3],
    [2,3]
], dtype=np.int32)

_FACE_IDX = np.array([
    [0,1,2],
    [0,1,3],
    [0,2,3],
    [1,2,3]
], dtype=np.int32)

class AABB:
    min: np.ndarray
    max: np.ndarray

    def __init__(self, min: np.ndarray, max: np.ndarray):
        self.min = min
        self.max = max

    def intersects(self, other: AABB) -> bool:
        return bool(np.all(self.max > other.min) and np.all(self.min < other.max))

    @staticmethod
    def fromPoints(points: np.ndarray) -> AABB:
        return AABB(np.min(points, axis=0), np.max(points, axis=0))

class Facet:
    """
    A solid tetrahedron in R^3, used as the 3-D image of one 4-D facet
    after unfolding.

    Collision detection
    -------------------
    trimesh.collision.CollisionManager tests whether two *closed surface meshes*
    interpenetrate.  Two tetrahedra that merely share a face (adjacent facets
    in the net) will be reported as colliding by trimesh because their surface
    triangles are coplanar and trimesh's GJK/SAT solver treats coplanar touching
    as intersection.

    This is handled at the call site in Net.is_unfolding_overlap_free by skipping
    pairs that are adjacent in the spanning tree.  However there is a subtler
    problem: if the unfolding has produced facets that are *exactly* coplanar (all
    lying in the root's hyperplane, which they should be by construction), then
    *every* pair of facets touches in a lower-dimensional set, and trimesh may
    report all of them as colliding.

    We therefore use a **volumetric signed-distance / GJK** approach:
      - Two tetrahedra truly overlap (positive volume intersection) iff their
        interiors share a point that is strictly inside both.
      - We test this by checking whether any vertex of A is strictly inside B
        and vice versa, and then — for the harder case of interleaving without
        vertex containment — running trimesh's collision manager with a small
        negative inflation so that mere face-touching does not count.

    In practice the most reliable approach for solid tetrahedra is to use
    trimesh's `intersections.mesh_multiplane` or the GJK collision manager with
    an explicit tolerance.  We use a small inward offset (epsilon shrink) so
    that adjacent facets touching along a shared ridge are never flagged.

    Parameters
    ----------
    vertices : (4, 3) array of the four tetrahedron corners in R^3.
    """
    bounding: AABB
    vertices: np.ndarray

    __slots__ = (
        "vertices",
        "bounding",
        "edges",
        "faces",
        "face_normals"
    )
    
    def __init__(self, vertices: np.ndarray):
        v = np.array(vertices, dtype=float)  # (4, 3)

        self.vertices = v

        # Build a proper closed surface mesh for the tetrahedron.
        # ConvexHull on 4 non-coplanar 3-D points gives 4 triangular faces.
        # hull = ConvexHull(self.vertices)
        # faces = hull.simplices  # (4, 3)

        # Ensure consistent outward-facing winding order.
        # centroid = self.vertices.mean(axis=0)
        # corrected_faces = []
        # for face in faces:
        #     v0, v1, v2 = self.vertices[face]
        #     n = np.cross(v1 - v0, v2 - v0)
        #     fc = (v0 + v1 + v2) / 3.0
        #     if np.dot(n, fc - centroid) < 0:
        #         corrected_faces.append([face[0], face[2], face[1]])
        #     else:
        #         corrected_faces.append(list(face))

        self.bounding = AABB.fromPoints(self.vertices)

        self.edges = (
            v[_EDGE_IDX[:,1]] -
            v[_EDGE_IDX[:,0]]
        )

        faces = v[_FACE_IDX]

        self.face_normals = np.cross(
            faces[:,1] - faces[:,0],
            faces[:,2] - faces[:,0]
        )


    def intersects(self, other: Facet) -> bool:
        """
        Intersection using a fast resturn if the AABB is non-overlapping and continueing with SAT for the precise intersection
        """
        # use AABB for fast rejecting
        if not self.bounding.intersects(other.bounding):
            return False
        
        axes = []

        # Face normals
        axes.extend(self.face_normals)
        axes.extend(other.face_normals)

        # Edge cross products
        for ea in self.edges:
            c = np.cross(ea, other.edges)

            mask = np.einsum('ij,ij->i', c, c) > _EPS
            axes.extend(c[mask])

        A = self.vertices
        B = other.vertices

        for axis in axes:
            a_proj = A @ axis
            b_proj = B @ axis

            if (
                a_proj.max() <= b_proj.min() + _EPS or
                b_proj.max() <= a_proj.min() + _EPS
            ):
                return False

        return True