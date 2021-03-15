import algopy
import numpy as np
from scipy.integrate import quad


def quad(f, a, b):
    n = 1000
    Q = np.linspace(a, b, n)
    W = 1 / n * np.ones(n)
    y = 0
    for w, q in zip(W, Q):
        y += f(q) * w
    return y


def eval_f(x):
    integrand = lambda t: x[0] * np.sin(x[1] * t)

    # return integrand(1.0)
    # print(integrand(1.0))
    return quad(integrand, 0, 10)


# STEP 1: trace the function evaluation
cg = algopy.CGraph()
x = algopy.Function([1, 2])
y = eval_f(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]

# STEP 2: use the computational graph to evaluate derivatives
print('gradient =', cg.gradient([1.0, 2.0]))
print('Jacobian =', cg.jacobian([1.0, 2.0]))
print('Hessian =', cg.hessian([1.0, 2.0]))
print('Hessian vector product =', cg.hess_vec([1.0, 2.0], [1.0, 0.0]))
