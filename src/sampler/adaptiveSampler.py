"""
Class to implement adaptive sampling heuristic based on
best error erstimates.
Author: Steffen Schotthöfer
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import scipy
from src.math import EntropyTools
from scipy.spatial import Delaunay


# import time


class AdaptiveSampler(object):
    # Defining members
    all_pts: np.ndarray  # shape = (nSize, nDim)
    all_grads: np.ndarray  # shape = (nSize, nDim)
    nDim: int
    nSize: int
    knn_param: int

    def __init__(self, points: np.ndarray = np.empty((0, 2,), dtype=float),
                 grads: np.ndarray = np.empty((0, 2), dtype=float), knn_param: int = 10):
        self.nDim = points.shape[1]
        self.nSize = points.shape[0]
        self.knn_param = knn_param
        self.all_pts = points
        self.all_grads = grads

    def sample_adative(self, poi: np.ndarray, max_diam: float, max_iter: int,
                       poi_grad: np.ndarray = np.zeros((1, 2))) -> bool:
        """
        brief: Samples adaptively, until the diameter of the polygon is smaller than max_diam, or until max_iter is reached.
               Adds all created points to the training data set.
        params:
                poi: point of interest. dim = (nDim)
                max_diam: maximal allowed diameter of alpha
                max_iter: maximum amount of iterations of the algorithm
        returns: True, if the diameter has been reached, else false
        """
        # define local variables
        et = EntropyTools(N=2)
        curr_vertices: np.ndarray
        curr_idx: np.ndarray = self.get_nearest_nbrs(poi)
        curr_pts: np.ndarray = self.all_pts[curr_idx]
        curr_grads: np.ndarray = self.all_grads[curr_idx]
        mean: np.ndarray
        diam: float = np.infty
        count: int = 0

        # run sampler
        while diam > max_diam and count < max_iter:

            # Compute the best polygon
            (curr_idx, curr_vertices, success) = self.compute_a(poi=poi, local_pts=curr_pts, local_grads=curr_grads,
                                                                global_idx=curr_idx)
            if not success:
                print("Error: Poi not in polygon in iteration: " + str(count))
                exit()
            diam = self.compute_diam_a(curr_vertices)
            # compute the geometric mean
            mean = np.mean(curr_vertices, axis=0).reshape((1, 2))  # .reshape((1, 2))
            # compute the new point u corresponding to the mean
            alpha_recons = et.reconstruct_alpha(et.convert_to_tensor_float(mean))
            u = et.reconstruct_u(alpha_recons).numpy()[:, 1:]
            # append new point to the training points (since we have already computed it)
            self.all_pts = np.append(self.all_pts, u, axis=0)
            self.all_grads = np.append(self.all_grads, mean, axis=0)
            self.nSize += 1
            # refine the current polygon
            curr_idx = np.append(curr_idx, [self.nSize - 1], axis=0)
            curr_pts = self.all_pts[curr_idx]
            curr_grads = self.all_grads[curr_idx]

            print("Current diameter: " + str(diam) + " in iter " + str(count))
            count += 1
            """
            plt.plot(curr_vertices[:, 0], curr_vertices[:, 1], 'o')
            plt.plot(poi_grad[0], poi_grad[1], '+')
            plt.plot(mean[0, 0], mean[0, 1], '*')
            plt.xlim([0.0, 0.3])
            plt.ylim([0.2, 0.45])
            plt.savefig("iter_" + str(count) + ".png", dpi=150)
            plt.clf()
            """
        return True

    def compute_a_wrapper(self, poi: np.ndarray) -> tuple:
        """
        searches nearest neighbors and computes the resulting best polygon.
        return: (vertices of best polygon w.r.t poi, success_flag)
        """
        curr_idx: np.ndarray = self.get_nearest_nbrs(poi)
        curr_pts: np.ndarray = self.all_pts[curr_idx]
        curr_grads: np.ndarray = self.all_grads[curr_idx]
        (_, vertices, success) = self.compute_a(poi, curr_pts, curr_grads, curr_idx)
        return vertices, success

    @staticmethod
    def compute_a(poi: np.ndarray, local_pts: np.ndarray, local_grads: np.ndarray, global_idx: np.ndarray) -> tuple:

        """
        Computes the best error approximation polygon A out of a local point cloud
        returns: (index_list,vertices)
                 index_list: indices of points of point cloud, that are used for the best polygon
                 vertices: coordinates of vertices that form the polygon.
        """
        n_size: int = len(local_pts)
        n_dim: int = local_pts.shape[1]
        local_pts_normal: np.ndarray
        local_pts_ortonormal: np.ndarray
        intersection_params: np.ndarray = np.zeros(shape=(n_size, n_size), dtype=float)  # param, where i intersects j
        vertices: np.ndarray = np.zeros(shape=(n_size, n_size, n_dim), dtype=float)  # intersection points from (i,j)
        wrong_side: np.ndarray = np.zeros(shape=(n_size, n_size), dtype=bool)
        index_list: list = []
        vertex_list: list = []

        # define helper functions
        def center_and_normalize() -> np.ndarray:
            """
            brief:  shifts the coordinate system s.t. poi = 0. normalizes the inputs
            """
            normals = local_pts - poi
            for i in range(0, n_size):
                normals[i, :] = normals[i, :] / np.linalg.norm(normals[i, :])
            return normals

        def compute_orto_complement() -> np.ndarray:
            """
            computes the orthogonal complement of the points
            """
            ortogonals = np.asarray([local_pts_normal[:, 1], -local_pts_normal[:, 0]]).T
            return ortogonals

        def compute_intersection(idx_i: int, idx_j: int) -> np.ndarray:
            """
            brief: Compute the intersection between the boundary conditions given by point i and j of the conv hull
            param : idx_i  = index of first point
                    idx_j = index of second point
            returns : [t_i, t_j] line parameters for boundary line i and j at the intersection point
            """
            # Compute linear system (only 2D right now)
            a = np.asarray([local_pts_ortonormal[idx_i], -1 * local_pts_ortonormal[idx_j]]).T
            b = - local_grads[idx_i] + local_grads[idx_j]
            # check if a is singular
            if np.linalg.matrix_rank(a) < n_dim:  # vectors are linearly dependent
                return np.asarray([np.nan, np.nan])
            else:
                return np.linalg.solve(a, b)

        def compute_vertex(idx_i: int, idx_j: int) -> np.ndarray:
            """
            Computes the coordinates of vertex between line i and j
            returns: vertex =  np.array(ndim)
            """
            if idx_i >= n_size or idx_j >= n_size:
                print("Index out of bounds")
                exit(1)
            if np.isnan(intersection_params[idx_i, idx_j]):
                return np.asarray([np.nan, np.nan])
            return local_grads[idx_i] + intersection_params[idx_i, idx_j] * local_pts_ortonormal[idx_i]

        def interior_of_hull(points: np.ndarray, x: np.ndarray):  # -> bool:
            """
             Checks if point is in the interior of a convex hull of points (currently in 2d)
            params: points: point cloud that forms the convex hull
                    x = query point
            return: True, if x is in the interiour of Conv(points)
                    False, if x is on the boundary or outside of Conv(points)
            """
            # catch case, when all points lie on one hyperplane
            ref_pt = points[0]
            try:
                hull = ConvexHull(points)
            except scipy.spatial.qhull.QhullError:
                print("Qhulll errror")
                return False
            boundary = points[hull.vertices]
            a: np.ndarray = np.zeros(x.size)  # edge vector
            b: np.ndarray = np.zeros(x.size)  # vector from first vertex to poi
            determinant: float
            inside: bool = True
            for i in range(0, len(boundary)):  # boundary points are sorted counter clockwise
                if i == len(boundary) - 1:
                    a = boundary[0] - boundary[i]
                    b = x - boundary[i]
                else:
                    a = boundary[i + 1] - boundary[i]
                    b = x - boundary[i]
                determinant = np.linalg.det([a.T, b.T])
                if determinant <= 0:  # point is on the left or directly on the boundary at this edge
                    inside = False
            return inside

        if not interior_of_hull(local_pts, poi):
            print("Point of interest not in interior of convex hull")
            return [], np.zeros(poi.shape), False

        local_pts_normal = center_and_normalize()
        local_pts_ortonormal = compute_orto_complement()

        for i in range(0, n_size):  # Compute the intersection points
            intersection_params[i, i] = np.nan
            vertices[i, i] = np.asarray([np.nan, np.nan])
            for j in range(i + 1, n_size):
                if i is not j:
                    t = compute_intersection(i, j)
                    intersection_params[i, j] = t[0]
                    intersection_params[j, i] = t[1]
                    vertices[i, j] = compute_vertex(i, j)
                    vertices[j, i] = vertices[i, j]

        # go over all edges, select the points on the right side.
        for i in range(0, n_size):  # self.nSize):
            for j in range(0, n_size):
                for k in range(0, n_size):
                    if i != j and i != k:  # if i == j, all points are on line.  set False, to stabilize algo
                        if j == k:  # diagonal is a line intersecting with itself. not applicable
                            wrong_side[j, k] = True
                        if np.isnan(intersection_params[j, k]):
                            wrong_side[j, k] = True
                        else:
                            if np.dot(vertices[j, k], local_pts_normal[i]) > np.dot(local_grads[i],
                                                                                    local_pts_normal[i]):
                                wrong_side[j, k] = True
                                wrong_side[k, j] = True
        for i in range(0, n_size):
            for j in range(0, n_size):
                if not wrong_side[i, j]:
                    if i not in index_list:
                        index_list.append(i)
                    if j > i:
                        vertex_list.append(vertices[i, j])
        return global_idx[index_list], np.asarray(vertex_list), True

    @staticmethod
    def compute_diam_a(used_vertices: np.ndarray) -> float:
        """
        Returns the diameter of the polygon spanned by all vertices of marked  in "used_vertices"
        returns: float: diam
        """
        diam: float = 0
        for i in range(0, len(used_vertices)):
            for j in range(0, len(used_vertices)):
                if i != j:
                    temp_diam = np.linalg.norm(used_vertices[i] - used_vertices[j])
                    if temp_diam > diam:
                        diam = temp_diam
        return diam

    @staticmethod
    def compute_boundary(poi: np.ndarray, pt: np.ndarray, grad: np.ndarray) -> np.ndarray:
        # normalize pt
        normal = pt - poi
        normal = normal / np.linalg.norm(normal)
        orto_normal = np.asarray([normal[1], -normal[0]])
        t = np.linspace(-2, 2, 100)
        res = np.zeros((100, 2))
        for j in range(0, 100):
            res[j] = grad + t[j] * orto_normal
        return res

    def get_nearest_nbrs(self, query_pt: np.ndarray) -> np.ndarray:
        """
        brief: compute the k nearest neighbors of pts out of the cloud self.pts, specified by self.knn_param
        param:
        """
        # elegant version (not compatible with numba):
        #       return np.argsort(np.linalg.norm(self.all_pts - query_pt, axis=1))[0:self.knn_param]
        t2 = np.linalg.norm(self.all_pts - query_pt, axis=1)
        # return np.argsort(t2)[0:self.knn_param]  # very costly!
        # get the first knn_param neighbors
        partitioned_idx = np.argpartition(t2, self.knn_param - 1)
        knn_distances = t2[partitioned_idx[0:self.knn_param]]
        knn_idx = np.argsort(knn_distances)
        return partitioned_idx[knn_idx]

    def compute_pois(self) -> np.ndarray:
        """
        Computes points of interest given a initial set of training data (self.pts, self.grads) by delaunay triangulation
        (We know the the worst points lie on edges between training points)
        """
        dim = self.all_pts.shape[1]
        tri = Delaunay(self.all_pts)

        plt.triplot(self.all_pts[:, 0], self.all_pts[:, 1], tri.simplices)
        plt.plot(self.all_pts[:, 0], self.all_pts[:, 1], 'o')
        plt.show()

        poi_list: list = []
        for simplex in tri.simplices:  # each side is the connection of all points exept one
            # find the geometric mean of each side
            for i in range(0, len(simplex)):
                mean = np.zeros(dim, dtype=float)
                for j in range(0, len(simplex)):
                    if j != i:
                        mean += self.all_pts[simplex[j]]
                mean /= float(len(simplex) - 1.0)
                poi_list.append(mean)

        return np.unique(np.asarray(poi_list), axis=0)  # only count each edge once


'''
def in_hull(points: np.ndarray, x: np.ndarray) -> bool:
    """
    brief: helper function, that checks, if x is in the convex hull of points 
    param: points = defining  the  convex hull(np  array, dim = (n_points x n_dim))
    x = query point(np array, dim = (n_dim))
    returns: true, if x in convex hull, else false
    """


    n_points: int = len(points)
    c: np.ndarray = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]  # concat
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success
'''
