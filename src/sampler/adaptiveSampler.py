"""
Class to implement adaptive sampling heuristic based on
best error erstimates.
Author: Steffen Schotthöfer
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from src.math import EntropyTools


def interior_of_hull(points: np.ndarray, x: np.ndarray):  # -> bool:
    """
     Checks if point is in the interior of a convex hull of points (currently in 2d)
    params: points: point cloud that forms the convex hull
            x = query point
    return: True, if x is in the interiour of Conv(points)
            False, if x is on the boundary or outside of Conv(points)
    """
    hull = ConvexHull(points)
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


def in_hull(points: np.ndarray, x: np.ndarray) -> bool:
    """
    brief: helper function, that checks, if x is in the convex hull of points
    param: points = defining the convex hull (np array, dim = (n_points x n_dim))
           x = query point  (np array, dim = (n_dim))
    returns: true, if x in convex hull, else false
    """
    n_points: int = len(points)
    c: np.ndarray = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]  # concat
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def funcV2(x: np.ndarray, gradx: np.ndarray, v0: float):
    # calculate boundary for triangle
    rhs = np.dot(x, gradx)
    return (rhs - x[0] * v0) / x[1]


class Refiner:
    normal_pts: np.ndarray
    grads: np.ndarray
    poi: np.ndarray

    def __init__(self, points: np.ndarray, grads: np.ndarray, poi: np.ndarray):
        self.pts = points
        self.grads = grads
        self.poi = poi

    def refine(self, new_point: np.ndarray, new_grad: np.ndarray):
        self.normal_pts = np.append(self.pts, new_point)
        self.grads = np.append(self.grads, new_grad)

        self.center_and_normalize()
        self.compute_orthogonal_complements()
        return 0

    def center_and_normalize(self):
        return 0


class AdaptiveSampler:
    # Defining members
    all_pts: np.ndarray  # shape = (nSize, nDim)
    all_grads: np.ndarray  # shape = (nSize, nDim)
    local_pts: np.ndarray  # shape = (knn_param, nDim)
    local_pts_normal: np.ndarray  # shape = (knn_param, nDim)
    local_pts_normal_orto: np.ndarray  # shape = (knn_param, nDim)
    local_grads: np.ndarray  # shape = (knn_param, nDim)
    nDim: int
    nSize: int
    intersection_params: np.ndarray  # shape = (knn_param,knn_param) intersection_params[i,j] = parametrization point of line i to cut line j
    vertices: np.ndarray  # shape = (knn_param,knn_param, nDim)
    vertices_used: np.ndarray  # shape = (knn_param, knn_param) in each row, maximum 2 entries can be positive! matrix is symmetric
    knn_param: int

    # members for adaptive sampling

    def __init__(self, points: np.ndarray = np.empty((0, 2,), dtype=float),
                 grads: np.ndarray = np.empty((0, 2), dtype=float), knn_param: int = 10):
        self.nDim = points.shape[1]
        self.nSize = points.shape[0]
        self.knn_param = knn_param
        self.all_pts = points
        self.all_grads = grads
        self.intersection_params = np.zeros((self.knn_param, self.knn_param), dtype=float)
        self.vertices = np.zeros((self.knn_param, self.knn_param, self.nDim), dtype=float)
        self.vertices_used = np.zeros((self.knn_param, self.knn_param), dtype=bool)
        self.local_pts_normal_orto = np.zeros((self.knn_param, self.nDim), dtype=float)
        self.local_pts_normal = np.zeros((self.knn_param, self.nDim), dtype=float)

    def sample_adative(self, poi: np.ndarray, max_diam: float, max_iter: int) -> bool:
        """
        brief: Samples adaptively, until the diameter of the polygon is smaller than max_diam, or until max_iter is reached.
               Adds all created points to the training data set.
        params:
                poi: point of interest. dim = (nDim)
                max_diam: maximal allowed diameter of alpha
                max_iter: maximum amount of iterations of the algorithm
        returns: True, if the diameter has been reached, else false
        """

        et = EntropyTools(N=2)
        curr_vertices: np.ndarray
        mean: float

        self.compute_a(poi)
        diam: float = self.compute_diam_a()
        while diam > max_diam:
            diam2 = diam
            curr_vertices = self.get_used_vertices()
            mean = np.mean(curr_vertices, axis=0).reshape((1, 2))
            alpha_recons = et.reconstruct_alpha(et.convert_to_tensorf(mean))
            u = et.reconstruct_u(alpha_recons).numpy()[:, 1:]
            self.all_pts = np.append(self.local_pts, u, axis=0)
            self.all_grads = np.append(self.local_grads, mean, axis=0)
            self.nSize += 1
            self.local_cloud_size += 1
            self.refine_a(poi)
            diam = self.compute_diam_a()
            print("h")
            print(diam)
        return True

    def refine_a(self, poi: np.ndarray, new_pt: np.ndarray, new_grad: np.ndarray) -> bool:
        """
               brief: refines the gradient polygon for given points and the point of interest
                       ==> saves them in vertex_used
               params: pts = samplingpoints
                       grads = gradients of sampling points  (dim = (n_points x n_dim))
                       poi =  point of interest. must be element of the convex hull of pts (np array, dim = (n_dim))
               returns: True, if completed succcessfully, else False
        """
        self.local_pts = np.append(self.get_used_vertices(), new_pt)
        refinement_grads = np.append()  # self.all_grads[nearest_indices]

        # Check, if poi is in the convex hull of pts
        if not interior_of_hull(self.local_pts, poi):
            print("Point of interest not in interior of convex hull")
            return False

        # normalize knn data
        self.center_and_normalize(poi)
        self.compute_orthogonal_complements()

        # Compute the intersection points
        for i in range(0, self.knn_param):
            self.intersection_params[i, i] = np.nan
            self.vertices[i, i] = [np.nan, np.nan]
            for j in range(i + 1, self.knn_param):
                if i is not j:
                    t = self.compute_intersection(i, j)
                    self.intersection_params[i, j] = t[0]
                    self.intersection_params[j, i] = t[1]
                    self.vertices[i, j] = self.compute_vertex(i, j)
                    self.vertices[j, i] = self.vertices[i, j]

        # go over all edges, select the points on the right side.
        wrong_side = np.zeros((self.knn_param, self.knn_param), dtype=bool)
        for i in range(0, self.knn_param):  # self.nSize):
            for j in range(0, self.knn_param):
                for k in range(0, self.knn_param):
                    if i != j and i != k:  # if i == j, all points are on line.  set False, to stabilize algo
                        if j == k:  # diagonal is a line intersecting with itself. not applicable
                            wrong_side[j, k] = True
                        if np.isnan(self.intersection_params[j, k]):
                            wrong_side[j, k] = True
                        else:
                            if np.dot(self.vertices[j, k], self.local_pts_normal[i]) > np.dot(self.local_grads[i],
                                                                                              self.local_pts_normal[i]):
                                wrong_side[j, k] = True
                                wrong_side[k, j] = True

        self.vertices_used = ~wrong_side

        return True

    def compute_a2(self, poi: np.ndarray, local_pts: np.ndarray, local_grads: np.ndarray) -> tuple:
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
        wrong_side: np.ndarray = np.zeros((self.knn_param, self.knn_param), dtype=bool)
        index_list: list = []

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
            if np.isnan(self.intersection_params[idx_i, idx_j]):
                return np.asarray([np.nan, np.nan])
            return local_grads[idx_i] + intersection_params[idx_i, idx_j] * local_pts_ortonormal[idx_i]

        if not interior_of_hull(self.local_pts, poi):
            print("Point of interest not in interior of convex hull")
            return [], np.empty()

        local_pts_normal = center_and_normalize()
        local_pts_ortonormal = compute_orto_complement()

        for i in range(0, n_size):  # Compute the intersection points
            intersection_params[i, i] = np.nan
            vertices[i, i] = [np.nan, np.nan]
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
                            if np.dot(self.vertices[j, k], self.local_pts_normal[i]) > np.dot(self.local_grads[i],
                                                                                              self.local_pts_normal[i]):
                                wrong_side[j, k] = True
                                wrong_side[k, j] = True
        for i in range(0, n_size):
            for j in range(i + 1, n_size):
                if not wrong_side[i, j]:
                    index_list.append(i)
        return index_list, vertices[~wrong_side]

    def compute_a(self, poi: np.ndarray) -> bool:
        """
        brief: Computes the gradient polygon for given points and the point of interest
                ==> saves them in vertex_used
        params: pts = samplingpoints
                grads = gradients of sampling points  (dim = (n_points x n_dim))
                poi =  point of interest. must be element of the convex hull of pts (np array, dim = (n_dim))
        returns: True, if completed succcessfully, else False
        """
        # Look only at the nearest neighbors (defined by knn_param)
        nearest_indices = self.get_nearest_nbrs(poi)
        self.local_pts = self.all_pts[nearest_indices]
        self.local_grads = self.all_grads[nearest_indices]

        # Check, if poi is in the convex hull of pts
        if not interior_of_hull(self.local_pts, poi):
            print("Point of interest not in interior of convex hull")
            return False

        # normalize knn data
        self.center_and_normalize(poi)
        self.compute_orthogonal_complements()

        # Compute the intersection points
        for i in range(0, self.knn_param):
            self.intersection_params[i, i] = np.nan
            self.vertices[i, i] = [np.nan, np.nan]
            for j in range(i + 1, self.knn_param):
                if i is not j:
                    t = self.compute_intersection(i, j)
                    self.intersection_params[i, j] = t[0]
                    self.intersection_params[j, i] = t[1]
                    self.vertices[i, j] = self.compute_vertex(i, j)
                    self.vertices[j, i] = self.vertices[i, j]

        # go over all edges, select the points on the right side.
        wrong_side = np.zeros((self.knn_param, self.knn_param), dtype=bool)
        for i in range(0, self.knn_param):  # self.nSize):
            for j in range(0, self.knn_param):
                for k in range(0, self.knn_param):
                    if i != j and i != k:  # if i == j, all points are on line.  set False, to stabilize algo
                        if j == k:  # diagonal is a line intersecting with itself. not applicable
                            wrong_side[j, k] = True
                        if np.isnan(self.intersection_params[j, k]):
                            wrong_side[j, k] = True
                        else:
                            if np.dot(self.vertices[j, k], self.local_pts_normal[i]) > np.dot(self.local_grads[i],
                                                                                              self.local_pts_normal[i]):
                                wrong_side[j, k] = True
                                wrong_side[k, j] = True
        self.vertices_used = ~wrong_side

        return True

    def compute_diam_a(self, used_vertices: np.ndarray) -> float:
        """
        Returns the diameter of the polygon spanned by all vertices of marked  in "vertices_used"
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

    def compute_vertex(self, i: int = 0, j: int = 1) -> np.ndarray:
        """
        Computes the coordinates of vertex between line i and j
        returns: vertex =  np.array(ndim)
        """
        if i > self.knn_param or j > self.knn_param:
            print("Index out of bounds")
            exit(1)
        if np.isnan(self.intersection_params[i, j]):
            return np.asarray([np.nan, np.nan])
        return self.local_grads[i] + self.intersection_params[i, j] * self.local_pts_normal_orto[i]

    def compute_boundary(self, i: int) -> np.ndarray:
        if i > self.knn_param:
            print("Index out of bounds")
            exit(1)

        t = np.linspace(-2, 2, 100)
        res = np.zeros((100, 2))
        for j in range(0, 100):
            res[j, :] = self.local_grads[i] + t[j] * self.local_pts_normal_orto[i]
        return res

    def compute_intersection(self, i: int = 0, j: int = 1) -> np.ndarray:
        """
        brief: Compute the intersection between the boundary conditions given by point i and j of the conv hull
        param : idx_i  = index of first point
                idx_j = index of second point
        returns : [t_i, t_j] line parameters for boundary line i and j at the intersection point
        """
        # Compute linear system (only 2D right now)
        # a0 = np.asarray([[self.pts_normal_orto[i][0], self.pts_normal_orto[j][0]],
        #                 [self.pts_normal_orto[i][1], self.pts_normal_orto[j][1]]])
        a = np.asarray([self.local_pts_normal_orto[i], -1 * self.local_pts_normal_orto[j]]).T
        b = - self.local_grads[i] + self.local_grads[j]
        # a2 = np.asarray([self.pts_normal_orto[j], self.pts_normal_orto[i]]).T

        # check if a is singular
        if np.linalg.matrix_rank(a) < self.nDim:
            # vectors are linearly dependent
            t = [np.nan, np.nan]
        else:
            t = np.linalg.solve(a, b)
            # t2 = np.linalg.solve(a2, b)
        # print(t2)
        # print(t)
        # print("___")
        # postprocessing (inf solutions)
        # todo

        return t

    def compute_orthogonal_complements(self) -> bool:
        """
        computes the orthogonal complement of the points
        """

        self.local_pts_normal_orto = np.copy(self.local_pts_normal)
        self.local_pts_normal_orto = np.asarray([self.local_pts_normal[:, 1], -self.local_pts_normal[:, 0]]).T
        # for comp in self.pts_normal_orto:
        #    comp = [comp[1], -comp[0]]
        # self.pts_normal_orto[:, 0] *= -1
        return True

    def center_and_normalize(self, poi: np.ndarray) -> bool:
        """
        brief:  shifts the coordinate system s.t. poi = 0. normalizes the inputs
        """
        self.local_pts_normal = self.local_pts - poi
        for i in range(0, self.knn_param):
            self.local_pts_normal[i, :] = self.local_pts_normal[i, :] / np.linalg.norm(self.local_pts_normal[i, :])

        return True

    def get_vertices(self) -> np.ndarray:
        res = []
        for i in range(0, self.knn_param):
            for j in range(i, self.knn_param):
                if not np.isnan(self.vertices[i, j, 0]):
                    res.append(self.vertices[i, j])

        return np.asarray(res)

    def get_used_vertices(self) -> np.ndarray:
        res = []
        for i in range(0, self.knn_param):
            for j in range(i + 1, self.knn_param):
                if self.vertices_used[i, j]:
                    res.append(self.vertices[i, j])

        return np.asarray(res)

    def get_nearest_nbrs(self, query_pt: np.ndarray) -> np.ndarray:
        """
        brief: compute the k nearest neighbors of pts out of the cloud self.pts, specified by self.knn_param
        param:
        """
        return np.argsort(np.linalg.norm(self.all_pts - query_pt, axis=1))[0:self.knn_param]


class adaptiveSamplerOld:
    # Defining members
    all_pts: np.ndarray  # shape = (nSize, nDim)
    all_grads: np.ndarray  # shape = (nSize, nDim)
    local_pts: np.ndarray  # shape = (knn_param, nDim)
    local_pts_normal: np.ndarray  # shape = (knn_param, nDim)
    local_pts_normal_orto: np.ndarray  # shape = (knn_param, nDim)
    local_grads: np.ndarray  # shape = (knn_param, nDim)
    nDim: int
    nSize: int
    intersection_params: np.ndarray  # shape = (knn_param,knn_param) intersection_params[i,j] = parametrization point of line i to cut line j
    vertices: np.ndarray  # shape = (knn_param,knn_param, nDim)
    vertices_used: np.ndarray  # shape = (knn_param, knn_param) in each row, maximum 2 entries can be positive! matrix is symmetric
    knn_param: int

    # members for adaptive sampling

    def __init__(self, points: np.ndarray = np.empty((0, 2,), dtype=float),
                 grads: np.ndarray = np.empty((0, 2), dtype=float), knn_param: int = 10):
        self.nDim = points.shape[1]
        self.nSize = points.shape[0]
        self.knn_param = knn_param
        self.all_pts = points
        self.all_grads = grads
        self.intersection_params = np.zeros((self.knn_param, self.knn_param), dtype=float)
        self.vertices = np.zeros((self.knn_param, self.knn_param, self.nDim), dtype=float)
        self.vertices_used = np.zeros((self.knn_param, self.knn_param), dtype=bool)
        self.local_pts_normal_orto = np.zeros((self.knn_param, self.nDim), dtype=float)
        self.local_pts_normal = np.zeros((self.knn_param, self.nDim), dtype=float)

    def compute_diam_a(self) -> float:
        """
        Returns the diameter of the polygon spanned by all vertices of marked  in "vertices_used"
        returns: float: diam
        """
        vertex_list: list = []
        diam: float = 0
        temp_diam: float = 0
        for i in range(0, self.knn_param):  # get all used vertices
            for j in range(i + 1, self.knn_param):  # look at upper tri diagonal matrix, to avoid dublicates
                if self.vertices_used[i, j]:
                    vertex_list.append(self.vertices[i, j])
        # diameter is the maxium distance between two nodes
        vertex_array: np.ndarray = np.asarray(vertex_list)
        for i in range(0, len(vertex_array)):
            for j in range(0, len(vertex_array)):
                if i != j:
                    temp_diam = np.linalg.norm(vertex_array[i] - vertex_array[j])
                    if temp_diam > diam:
                        diam = temp_diam
        return diam

    def compute_a(self, poi: np.ndarray) -> bool:
        """
        brief: Computes the gradient polygon for given points and the point of interest
                ==> saves them in vertex_used
        params: pts = samplingpoints
                grads = gradients of sampling points  (dim = (n_points x n_dim))
                poi =  point of interest. must be element of the convex hull of pts (np array, dim = (n_dim))
        returns: True, if completed succcessfully, else False
        """
        # Look only at the nearest neighbors (defined by knn_param)
        nearest_indices = self.get_nearest_nbrs(poi)
        self.local_pts = self.all_pts[nearest_indices]
        self.local_grads = self.all_grads[nearest_indices]

        # Check, if poi is in the convex hull of pts
        if not interior_of_hull(self.local_pts, poi):
            print("Point of interest not in interior of convex hull")
            return False

        # normalize knn data
        self.center_and_normalize(poi)
        self.compute_orthogonal_complements()

        # Compute the intersection points
        for i in range(0, self.knn_param):
            self.intersection_params[i, i] = np.nan
            self.vertices[i, i] = [np.nan, np.nan]
            for j in range(i + 1, self.knn_param):
                if i is not j:
                    t = self.compute_intersection(i, j)
                    self.intersection_params[i, j] = t[0]
                    self.intersection_params[j, i] = t[1]
                    self.vertices[i, j] = self.compute_vertex(i, j)
                    self.vertices[j, i] = self.vertices[i, j]

        # go over all edges, select the points on the right side.
        wrong_side = np.zeros((self.knn_param, self.knn_param), dtype=bool)
        for i in range(0, self.knn_param):  # self.nSize):
            for j in range(0, self.knn_param):
                for k in range(0, self.knn_param):
                    if i != j and i != k:  # if i == j, all points are on line.  set False, to stabilize algo
                        if j == k:  # diagonal is a line intersecting with itself. not applicable
                            wrong_side[j, k] = True
                        if np.isnan(self.intersection_params[j, k]):
                            wrong_side[j, k] = True
                        else:
                            if np.dot(self.vertices[j, k], self.local_pts_normal[i]) > np.dot(self.local_grads[i],
                                                                                              self.local_pts_normal[i]):
                                wrong_side[j, k] = True
                                wrong_side[k, j] = True
        self.vertices_used = ~wrong_side

        return True

    def compute_vertex(self, i: int = 0, j: int = 1) -> np.ndarray:
        """
        Computes the coordinates of vertex between line i and j
        returns: vertex =  np.array(ndim)
        """
        if i > self.knn_param or j > self.knn_param:
            print("Index out of bounds")
            exit(1)
        if np.isnan(self.intersection_params[i, j]):
            return np.asarray([np.nan, np.nan])
        return self.local_grads[i] + self.intersection_params[i, j] * self.local_pts_normal_orto[i]

    def compute_boundary(self, i: int) -> np.ndarray:
        if i > self.knn_param:
            print("Index out of bounds")
            exit(1)

        t = np.linspace(-2, 2, 100)
        res = np.zeros((100, 2))
        for j in range(0, 100):
            res[j, :] = self.local_grads[i] + t[j] * self.local_pts_normal_orto[i]
        return res

    def compute_intersection(self, i: int = 0, j: int = 1) -> np.ndarray:
        """
        brief: Compute the intersection between the boundary conditions given by point i and j of the conv hull
        param : idx_i  = index of first point
                idx_j = index of second point
        returns : [t_i, t_j] line parameters for boundary line i and j at the intersection point
        """
        # Compute linear system (only 2D right now)
        # a0 = np.asarray([[self.pts_normal_orto[i][0], self.pts_normal_orto[j][0]],
        #                 [self.pts_normal_orto[i][1], self.pts_normal_orto[j][1]]])
        a = np.asarray([self.local_pts_normal_orto[i], -1 * self.local_pts_normal_orto[j]]).T
        b = - self.local_grads[i] + self.local_grads[j]
        # a2 = np.asarray([self.pts_normal_orto[j], self.pts_normal_orto[i]]).T

        # check if a is singular
        if np.linalg.matrix_rank(a) < self.nDim:
            # vectors are linearly dependent
            t = [np.nan, np.nan]
        else:
            t = np.linalg.solve(a, b)
            # t2 = np.linalg.solve(a2, b)
        # print(t2)
        # print(t)
        # print("___")
        # postprocessing (inf solutions)
        # todo

        return t

    def compute_orthogonal_complements(self) -> bool:
        """
        computes the orthogonal complement of the points
        """

        self.local_pts_normal_orto = np.copy(self.local_pts_normal)
        self.local_pts_normal_orto = np.asarray([self.local_pts_normal[:, 1], -self.local_pts_normal[:, 0]]).T
        # for comp in self.pts_normal_orto:
        #    comp = [comp[1], -comp[0]]
        # self.pts_normal_orto[:, 0] *= -1
        return True

    def center_and_normalize(self, poi: np.ndarray) -> bool:
        """
        brief:  shifts the coordinate system s.t. poi = 0. normalizes the inputs
        """
        self.local_pts_normal = self.local_pts - poi
        for i in range(0, self.knn_param):
            self.local_pts_normal[i, :] = self.local_pts_normal[i, :] / np.linalg.norm(self.local_pts_normal[i, :])

        return True

    def get_vertices(self) -> np.ndarray:
        res = []
        for i in range(0, self.knn_param):
            for j in range(i, self.knn_param):
                if not np.isnan(self.vertices[i, j, 0]):
                    res.append(self.vertices[i, j])

        return np.asarray(res)

    def get_used_vertices(self) -> np.ndarray:
        res = []
        for i in range(0, self.knn_param):
            for j in range(i + 1, self.knn_param):
                if self.vertices_used[i, j]:
                    res.append(self.vertices[i, j])

        return np.asarray(res)

    def get_nearest_nbrs(self, query_pt: np.ndarray) -> np.ndarray:
        """
        brief: compute the k nearest neighbors of pts out of the cloud self.pts, specified by self.knn_param
        param:
        """
        return np.argsort(np.linalg.norm(self.all_pts - query_pt, axis=1))[0:self.knn_param]
