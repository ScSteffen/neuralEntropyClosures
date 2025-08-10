import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special

# ----------------------------
# Quadrature + basis utilities
# ----------------------------


def qGaussLegendre2D(degree: int):
    pts_1d, w_1d = np.polynomial.legendre.leggauss(degree)
    pts = np.array([[x, y] for x in pts_1d for y in pts_1d])
    weights = np.array([wx * wy for wx in w_1d for wy in w_1d])
    return pts, weights


def compute_spherical_harmonics_2D(
    mu: np.ndarray, phi: np.ndarray, degree: int
) -> np.ndarray:
    input_dim_dict_2D = {1: 3, 2: 6, 3: 10, 4: 15, 5: 21}
    n_system = input_dim_dict_2D[degree]
    sh_basis = np.zeros((n_system, len(mu)))

    count = 0
    for l in range(degree + 1):
        for k in range(-l, l + 1):
            if (k + l) % 2 == 0:
                for i in range(len(mu)):
                    if k < 0:
                        Y = scipy.special.sph_harm(abs(k), l, phi[i], np.arccos(mu[i]))
                        Y = np.sqrt(2) * (-1) ** (k + l) * Y.imag
                    elif k > 0:
                        Y = scipy.special.sph_harm(k, l, phi[i], np.arccos(mu[i]))
                        Y = np.sqrt(2) * (-1) ** (k + l) * Y.real
                    else:
                        Y = scipy.special.sph_harm(k, l, phi[i], np.arccos(mu[i]))
                        Y = Y.real
                    sh_basis[count, i] = Y
                count += 1
    return sh_basis


# -----------------
# ICNN Convex Layer
# -----------------


class ConvexLayer(nn.Module):
    def __init__(self, input_dim, z_dim, use_nonneg=True):
        super().__init__()
        # Dense layer on z with non-negative weights if use_nonneg
        self.z_layer = nn.Linear(z_dim, z_dim, bias=False)
        # if use_nonneg:
        nn.init.uniform_(self.z_layer.weight, 0, 0.01)
        # self.z_layer.weight.data.clamp_(0)  # enforce non-neg at start
        # Dense layer on x (input)
        self.x_layer = nn.Linear(input_dim, z_dim, bias=True)
        # nn.init.uniform_(self.x_layer.weight, -0.5, 0.5)

    def forward(self, z, x):
        # Enforce non-negative weights for z_layer by clamp (soft projection)
        self.z_layer.weight.data.clamp_(0)

        # z_out = z @ torch.exp(self.z_layer.weight.T)
        z_out = self.z_layer(z)
        x_out = self.x_layer(x)
        out = z_out + x_out
        return out


class ConvexOutputLayer(nn.Module):
    def __init__(self, input_dim, z_dim):
        super().__init__()
        self.z_layer = nn.Linear(z_dim, 1, False)
        nn.init.uniform_(self.z_layer.weight, 0, 0.01)
        self.x_layer = nn.Linear(input_dim, 1, bias=True)
        nn.init.uniform_(self.x_layer.weight, -0.5, 0.5)
        self.ln_in = nn.LayerNorm(z_dim)

    def forward(self, z, x):
        z = self.ln_in(z)
        self.z_layer.weight.data.clamp_(0)
        z_out = self.z_layer(z)
        # z_out = z @ torch.exp(self.z_layer.weight.T)
        x_out = self.x_layer(x)

        out = z_out + x_out
        return out


# --------------
# MK11 Core Model
# --------------


class MK11CoreModel(nn.Module):
    def __init__(self, input_dim, width, depth):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.depth = depth

        self.input_layer = nn.Linear(input_dim, width)
        nn.init.uniform_(self.input_layer.weight, -0.5, 0.5)
        self.convex_layers = nn.ModuleList(
            [ConvexLayer(input_dim, width) for _ in range(depth)]
        )
        self.output_layer = ConvexOutputLayer(input_dim, width)
        self.relu = nn.ELU()

    def forward(self, x):
        z = self.relu(self.input_layer(x))
        for layer in self.convex_layers:
            z = self.relu(layer(z, x)) + z
        out = self.relu(self.output_layer(z, x))
        return out


# -----------------------
# Sobolev Wrapper for MK11
# -----------------------


class MK11SobolevWrapper(nn.Module):
    def __init__(
        self,
        core_model: nn.Module,
        polynomial_degree=1,
        gamma=0.0,
    ):
        super().__init__()

        self.core_model = core_model
        self.poly_degree = polynomial_degree
        self.regularization_gamma = gamma

    def forward(self, x):
        x.requires_grad_(True)
        h = self.core_model(x)
        alpha = torch.autograd.grad(
            h, x, grad_outputs=torch.ones_like(h), create_graph=True
        )[0]
        return h, alpha

    def reconstruct_alpha(self, alpha):
        alpha_clamped = torch.clamp(alpha, -50, 50)
        exp_term = torch.exp(torch.matmul(alpha_clamped, self.moment_basis[1:, :]))
        weighted_sum = torch.matmul(exp_term, self.quad_weights.T).squeeze(-1)
        alpha_0 = -(
            torch.log(self.moment_basis[0, 0]) + torch.log(weighted_sum)
        ).unsqueeze(1)
        return torch.cat([alpha_0, alpha], dim=1)

    def reconstruct_u(self, alpha):
        alpha_clamped = torch.clamp(alpha, -50, 50)
        f_quad = torch.exp(torch.matmul(alpha_clamped, self.moment_basis))
        weighted_f = f_quad * self.quad_weights
        u_rec = torch.matmul(weighted_f, self.moment_basis.T)
        alpha_reg = self.regularization_gamma_vector * alpha
        return u_rec + alpha_reg


# ------------
# Full MK11 Model
# ------------


class MK11(nn.Module):
    def __init__(
        self,
        polynomial_degree=2,
        width=10,
        depth=4,
        gamma=0.0,
    ):
        super().__init__()
        # Build core model input_dim from spherical harmonics dimension
        input_dim_dict_2D = {1: 3, 2: 6, 3: 10, 4: 15, 5: 21}
        input_dim = input_dim_dict_2D[polynomial_degree]

        self.core_model = MK11CoreModel(input_dim - 1, width, depth)
        self.sobolev_wrapper = MK11SobolevWrapper(
            self.core_model,
            polynomial_degree=polynomial_degree,
            gamma=gamma,
        )

    def forward(self, x):
        return self.sobolev_wrapper(x)
        # return self.core_model(x)
