from __future__ import annotations
import numpy as np
from scipy.spatial import ConvexHull
import trimesh

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
        min = np.full(len(points[0]), np.inf)
        max = np.full(len(points[0]), -np.inf)
        for point in points:
            for i, value in enumerate(point):
                if min[i] > value:
                    min[i] = value
                if max[i] < value:
                    max[i] = value

        return AABB(min, max)

class Facet:
    mesh: trimesh.Trimesh
    bounding: AABB

    def __init__(self, vertices: np.ndarray):
        hull = ConvexHull(vertices)
        points = vertices
        faces = hull.simplices

        # Build trimesh object
        self.mesh = trimesh.Trimesh(
            vertices=points,
            faces=faces,
            process=False
        )
        
        self.bounding = AABB.fromPoints(vertices)

    def intersects(self, other: Facet) -> bool:
        if not self.bounding.intersects(other.bounding):
            return False
        
        manager = trimesh.collision.CollisionManager()
        manager.add_object("self", self.mesh)
        return bool(manager.in_collision_single(other.mesh))