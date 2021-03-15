from numpy import array, sqrt
from numpy.linalg import norm


def octahedron():
    """ Get the vertices and faces for an octahedron that
    fits into the unit sphere. """

    vertices = array(
        [
            [0.0, 0.0, +1.0],
            [0.0, +1.0, 0.0],
            [+1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    ).transpose()

    faces = (
        array(
            [
                [1, 2, 3],
                [1, 2, 6],
                [1, 3, 5],
                [1, 5, 6],
                [4, 5, 6],
                [4, 2, 6],
                [4, 3, 5],
                [4, 2, 3],
            ]
        ).transpose()
        - 1
    )

    return vertices, faces


def icosahedron():
    """ Get the vertices and faces for an icosahedron that
    fits into the unit sphere. """

    r = (1 + sqrt(5)) / 2.0  # golden ratio
    vertices = array(
        [
            [0, 1, r],
            [0, 1, -r],
            [0, -1, r],
            [0, -1, -r],
            [1, r, 0],
            [1, -r, 0],
            [-1, r, 0],
            [-1, -r, 0],
            [r, 0, 1],
            [r, 0, -1],
            [-r, 0, 1],
            [-r, 0, -1],
        ]
    ).transpose()
    # normalize points to live on the unit sphere
    for i in range(12):
        vertices[:, i] /= norm(vertices[:, i])

    faces = (
        array(
            [
                [1, 3, 9],
                [1, 3, 11],
                [1, 5, 7],
                [1, 7, 11],
                [2, 5, 7],
                [2, 12, 7],
                [2, 5, 10],
                [2, 4, 10],
                [2, 4, 12],
                [11, 8, 12],
                [4, 8, 12],
                [4, 8, 6],
                [4, 6, 10],
                [6, 10, 9],
                [5, 10, 9],
                [1, 9, 5],
                [7, 11, 12],
                [3, 9, 6],
                [3, 8, 6],
                [3, 8, 11],
            ]
        ).transpose()
        - 1
    )

    return vertices, faces
