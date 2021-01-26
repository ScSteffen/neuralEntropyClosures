from numpy import sqrt, ones
from numpy.linalg import norm
from scipy.optimize import least_squares, minimize

from sphericalquadpy.ldfeLinDisc.ldfe import computeweights, computeareas, \
    computeomegas, f, optimizeposition

a = 1 / sqrt(3)
rhos = 0.1 * ones(4)
# x0, x1, z0, z1 = 0.0, 0.1, 0.0, 0.1
# omegas = computeomegas(x0, x1, z0, z1)
# areas = computeareas(omegas, x0, x1, z0, z1)

# eps = 0.001
# v = [0 + eps, 1 - eps]
# for a in v:
#    for b in v:
#        for c in v:
#            for d in v:
#                rhos = [a, b, c, d]
#                weights = computeweights(rhos, omegas, a, x0, x1, z0, z1)
#                print(norm(weights - areas))

# y = f(rhos, omegas, a, x0, x1, z0, z1, areas)
# print(y)
# print(areas)
# print(weights)

x0, x1, z0, z1 = 0.0, 0.1, 0.0, 0.1
omegas = computeomegas(x0, x1, z0, z1)
areas = computeareas(omegas, x0, x1, z0, z1)
weights = computeweights(rhos, omegas, a, x0, x1, z0, z1)
#print(weights)


def tomin(r):
    #print("###")

    #print(r)
    y = f(r, omegas, a, x0, x1, z0, z1, areas)
    #print(y)
    return norm(y)


res = least_squares(tomin, rhos, bounds=((0, 1)))
print(res)
print(res.x)
res = optimizeposition(areas, omegas, x0, x1, z0, z1)
print(res)

res = minimize(tomin, rhos, bounds=((0, 1), (0, 1), (0, 1), (0, 1)))
print(res)

optimizeposition(areas, omegas, x0, x1, z0, z1)
