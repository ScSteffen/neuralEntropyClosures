"""
Class to implement adaptive sampling heuristic based on
best error erstimates.
Author: Steffen Schotthöfer
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


def in_hull(points, x):
    """
    brief: helper function, that checks, if x is in the convex hull of points
    param: points = defining the convex hull (np array, dim = (n_points x n_dim))
           x = query point  (np array, dim = (n_dim))
    returns: true, if x in convex hull, else false
    """
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]  # concat
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def funcV2(x, gradx, v0):
    # calculate boundary for triangle
    rhs = np.dot(x, gradx)
    return (rhs - x[0] * v0) / x[1]


class adaptiveSampler:
    # Defining members
    pts: np.ndarray
    pts_normal: np.ndarray
    pts_normal_orto: np.ndarray
    grads: np.ndarray
    # fct_values: np.ndarray
    dim: int
    nSize: int
    intersection_params: np.ndarray  # intersection_params[i,j] = parametrization point of line i to cut line j
    vertices: np.ndarray
    vertices_used: np.ndarray  # in each row, maximum 2 entries can be positive! matrix is symmetric

    def __init__(self, nSize=10, nDim=2, points=np.empty((0, 2,), dtype=float), grads=np.empty((0, 2), dtype=float)):
        self.dim = nDim
        self.nSize = nSize
        self.pts = points
        self.grads = grads
        self.intersection_params = np.zeros((nSize, nSize), dtype=float)
        self.vertices = np.zeros((nSize, nSize, nDim), dtype=float)
        self.vertices_used = np.zeros((nSize, nSize), dtype=bool)
        self.pts_normal_orto = np.zeros((nSize, nDim), dtype=float)

    def compute_a(self, poi, poiGrad) -> np.ndarray:
        """
        brief: Computes the gradient polygon for given points and the point of interest
        params: pts = samplingpoints
                grads = gradients of sampling points  (dim = (n_points x n_dim))
                poi =  point of interest. must be element of the convex hull of pts (np array, dim = (n_dim))
        returns: vert = vertices of the gradient polygon
        """

        # Check, if poi is in the convex hull of pts
        if not in_hull(self.pts, poi):
            print("Point of interest not in convex hull")
            exit(1)

        # normalize data
        self.center_and_normalize(poi)
        self.compute_orthogonal_complements()

        # Compute the intersection points
        for i in range(0, self.nSize):
            self.intersection_params[i, i] = np.nan
            self.vertices[i, i] = [np.nan, np.nan]
            for j in range(i + 1, self.nSize):
                if i is not j:
                    t = self.compute_intersection(i, j)
                    self.intersection_params[i, j] = t[0]
                    self.intersection_params[j, i] = t[1]
                    self.vertices[i, j] = self.compute_vertex(i, j)
                    self.vertices[j, i] = self.vertices[i, j]

        # go over all edges, select the points on the right side.
        wrongSide = np.zeros((self.nSize, self.nSize), dtype=bool)
        for i in range(0, self.nSize):  # self.nSize):
            for j in range(0, self.nSize):
                for k in range(0, self.nSize):
                    if i != j and i != k:  # if i == j, all points are on line.  set False, to stabilize algo
                        if j == k:
                            wrongSide[j, k] = True
                        if not np.isnan(self.intersection_params[j, k]):
                            if np.dot(self.vertices[j, k], self.pts_normal[i]) > np.dot(self.grads[i],
                                                                                        self.pts_normal[i]):
                                wrongSide[j, k] = True
                                wrongSide[k, j] = True

        self.vertices_used = ~wrongSide
        # starting from the first three points, check if we can reduce diam(a)

        return self.vertices_used

    def compute_vertex(self, i=0, j=1) -> np.ndarray:
        """
        Computes the coordinates of vertex between line i and j
        returns: vertex =  np.array(ndim)
        """
        if np.isnan(self.intersection_params[i, j]):
            return np.asarray([np.nan, np.nan])
        return self.grads[i] + self.intersection_params[i, j] * self.pts_normal_orto[i]

    def compute_boundary(self, i):
        t = np.linspace(-2, 2, 100)
        res = np.zeros((100, 2))
        for j in range(0, 100):
            res[j, :] = self.grads[i] + t[j] * self.pts_normal_orto[i]
        return res

    def compute_intersection(self, i=0, j=1) -> np.ndarray:
        """
        brief: Compute the intersection between the boundary conditions given by point i and j of the conv hull
        param : idx_i  = index of first point
                idx_j = index of second point
        returns : [t_i, t_j] line parameters for boundary line i and j at the intersection point
        """
        # Compute linear system (only 2D right now)
        # a0 = np.asarray([[self.pts_normal_orto[i][0], self.pts_normal_orto[j][0]],
        #                 [self.pts_normal_orto[i][1], self.pts_normal_orto[j][1]]])
        a = np.asarray([self.pts_normal_orto[i], -1 * self.pts_normal_orto[j]]).T
        b = - self.grads[i] + self.grads[j]
        # a2 = np.asarray([self.pts_normal_orto[j], self.pts_normal_orto[i]]).T

        # check if a is singular
        if np.linalg.matrix_rank(a) < self.dim:
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

        self.pts_normal_orto = np.copy(self.pts_normal)
        self.pts_normal_orto = np.asarray([self.pts_normal[:, 1], -self.pts_normal[:, 0]]).T
        # for comp in self.pts_normal_orto:
        #    comp = [comp[1], -comp[0]]
        # self.pts_normal_orto[:, 0] *= -1
        return True

    def center_and_normalize(self, poi) -> bool:
        """
        brief:  shifts the coordinate system s.t. poi = 0. normalizes the inputs
        """
        self.pts_normal = self.pts - poi
        for i in range(0, self.nSize):
            self.pts_normal[i, :] = self.pts_normal[i, :] / np.linalg.norm(self.pts_normal[i, :])

        return True

    def get_vertices(self) -> np.ndarray:
        res = []
        for i in range(0, self.nSize):
            for j in range(i, self.nSize):
                if not np.isnan(self.vertices[i, j, 0]):
                    res.append(self.vertices[i, j])

        return np.asarray(res)

    def get_used_vertices(self) -> np.ndarray:
        res = []
        for i in range(0, self.nSize):
            for j in range(i + 1, self.nSize):
                if self.vertices_used[i, j]:
                    res.append(self.vertices[i, j])

        return np.asarray(res)
