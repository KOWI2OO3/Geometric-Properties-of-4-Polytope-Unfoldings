from typing import Callable
import numpy.typing as npt

from scipy.spatial import ConvexHull
import numpy as np

def generate_random_point(n: int, dim : int = 4):
    return np.random.default_rng().random((n, dim))

def random_unit_vector(d=4):
    x = np.random.normal(size=d)
    return x / np.linalg.norm(x)

# Generates points from the distribution and the number of points
# The distribution is used for xyzw of the point 
def generate_unit_points(distribution: Callable[[], npt.NDArray[np.float64]], n: int, points : list[npt.NDArray[np.float64]] | None = None) -> list[npt.NDArray[np.float64]]:
    if points is None:
        points = []

    for _ in range(0, n):
        point = distribution()
        
        normalized = point / np.linalg.norm(point)
        points.append(normalized);
    return points


def generate_uniform_unit_points(n: int, dim : int = 4) -> list[npt.NDArray[np.float64]]:
    return generate_unit_points(lambda : np.random.default_rng().normal(size=dim), n)

# Sampling using a gaussian distribution, which will result in uniform sampling in a sphere
def generate_uniform_unit_convex(n: int, dim : int = 4) -> ConvexHull:
    return ConvexHull(generate_uniform_unit_points(n, dim))
 
# n : number of total points
# p : Ratio of clustering [0 - 1] (0: no clustering, 1: max clustering)
# dim : dimensions to generate in
# sphere_point_ratio : that ratio of spherical points vs clustering points
#   - required as otherwise all points would be in the cluster and no proper shape would be formed
# min_clustering_size : defines the variance of the normal distribution used to sample points in the cluster
def generate_clustering_unit_convex(n: int, p : float, dim : int = 4, sphere_point_ratio : float = 1/6, min_clustering_size: float = 0.2) -> ConvexHull:
    inter = n * (1 - sphere_point_ratio)

    sphere = generate_uniform_unit_points(int(n * sphere_point_ratio + inter * (1 - p)), dim)
    centre = random_unit_vector(dim)

    return ConvexHull(generate_unit_points(lambda : centre + np.random.default_rng().normal(scale=min_clustering_size), int(inter * p), sphere))

def generate_convext_with_deformation(n: int , deformation: npt.NDArray[np.float64], dim : int = 4) -> ConvexHull:
    return ConvexHull(generate_uniform_unit_points(n, dim) * deformation)