import math

import torch


def inverse_sigmoid(value, min_val, max_val):
    """Map a bounded scalar to the raw parameter used before the sigmoid."""

    p = (value - min_val) / (max_val - min_val)
    p = max(min(p, 1 - 1e-6), 1e-6)
    return math.log(p / (1 - p))


def map_sigmoid_to_range(raw_value, min_val, max_val):
    """Map a raw trainable scalar to the bounded physical interval."""

    return min_val + torch.sigmoid(raw_value) * (max_val - min_val)


def plane_stress_stress_components(du_dx, dv_dy, du_dy, dv_dx, youngs_modulus, poisson_ratio):
    """Return the plane-stress constitutive response used by the main script."""

    c_matrix_scale = youngs_modulus / (1 - poisson_ratio**2)
    shear_modulus = youngs_modulus / (2 * (1 + poisson_ratio))
    sigma_xx = c_matrix_scale * (du_dx + poisson_ratio * dv_dy)
    sigma_yy = c_matrix_scale * (poisson_ratio * du_dx + dv_dy)
    sigma_xy = shear_modulus * (du_dy + dv_dx)
    return sigma_xx, sigma_xy, sigma_yy
