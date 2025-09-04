import numpy as np
from scipy.fft import fft2, ifft2
import math
import tensorflow as tf
from config import DEFAULT_R0, DEFAULT_LAMBDA, DEFAULT_DX, DEFAULT_DY, DEFAULT_N, CONST

def density_contrast(z, r0=DEFAULT_R0, lambda_=DEFAULT_LAMBDA, const=CONST):
    """
    New density contrast function (kg/m^3).
    z : depth (km)
    """
    return (const + r0 * np.exp(-lambda_ * (z / 1000.0))) #* 1000.0


def FW_Granser(z, r0, lambda_, nx, ny, dx, dy, n, const):
    """Forward gravity model using new Δρ(z) formulation"""
    z0_val = (np.max(z) - np.min(z)) / 2.0
    # nx, ny = z.shape[1], z.shape[0]
    nx0_orig, ny0_orig = nx, ny

    # First extension
    new_nx1 = nx + nx // 2
    new_ny1 = ny + ny // 2
    z1 = np.zeros((new_ny1, new_nx1))
    z1[:ny, :nx] = z
    z1[0, nx + nx // 2 - 1] = 0
    z1[ny + ny // 2 - 1, 0] = 0
    z1 = np.rot90(z1, 2)

    # Second extension
    new_nx2 = nx + 2 * (nx // 2)
    new_ny2 = ny + 2 * (ny // 2)
    z2 = np.zeros((new_ny2, new_nx2))
    z2[:z1.shape[0], :z1.shape[1]] = z1
    z2[0, new_nx2 - 1] = 0
    z2[new_ny2 - 1, 0] = 0
    z2 = np.rot90(z2, 2)

    # Adjust dimensions to even
    if nx % 2 != 0:
        nx = nx - 1
        z2 = z2[:, :-1]
    if ny % 2 != 0:
        ny = ny - 1
        z2 = z2[:-1, :]

    nxm = 2 * nx
    nym = 2 * ny
    h = z2 - z0_val   # relative depth perturbation

    # Wave number setup
    dkx = 2 * np.pi / ((nxm - 1) * dx)
    dky = 2 * np.pi / ((nym - 1) * dy)
    kx = np.zeros(nxm)
    ky = np.zeros(nym)
    nyqx = nxm // 2 + 1
    nyqy = nym // 2 + 1

    for j in range(nxm):
        if j < nyqx:
            kx[j] = j * dkx
        else:
            kx[j] = (j - nxm) * dkx

    for i in range(nym):
        if i < nyqy:
            ky[i] = i * dky
        else:
            ky[i] = (i - nym) * dky

    Ky, Kx = np.meshgrid(ky, kx, indexing='ij')
    k_grid = np.sqrt(Kx**2 + Ky**2)

    # --- Apply new Δρ(z) directly ---
    rho_contrast = density_contrast(h, r0, lambda_, const)

    # Fourier transform of density contrast
    tongF = fft2(rho_contrast)

    # Optional series expansion terms (kept for consistency)
    for m in range(1, n + 1):
        term = ((-k_grid) ** m) / math.factorial(m)
        F_term = fft2(rho_contrast * (h ** m))
        tongF = tongF - term * F_term

    # Gravitational effect in Fourier domain
    Fg = tongF / (k_grid + 1e-12)  # avoid divide-by-zero
    g0 = ifft2(Fg)
    r_g0 = np.real(g0)

    # Scaling constant (mGal units)
    hs1 = 2 * np.pi * 20 / 3
    g1 = r_g0 * hs1

    # Extract central region
    start_row = ny // 2
    start_col = nx // 2
    g1_cropped = g1[start_row:start_row + ny0_orig, start_col:start_col + nx0_orig]

    # Constant background term from mean density contrast at z0
    const_term = np.mean(density_contrast(z0_val, r0, lambda_, const))
    g = g1_cropped + const_term + 1e5

    return g

def calculate_gravity_field(depth_map, r0=DEFAULT_R0, lambda_=DEFAULT_LAMBDA,
                          dx=DEFAULT_DX, dy=DEFAULT_DY, n=DEFAULT_N, const=CONST):
    """
    TensorFlow-compatible wrapper for FW_Granser that handles batches
    Args:
        depth_map: tensor of shape (batch_size, height, width, 1)
    Returns:
        gravity_field: tensor of shape (batch_size, height, width, 1)
    """
    # Get shape information
    batch_size = tf.shape(depth_map)[0]
    nx = depth_map.shape[2]
    ny = depth_map.shape[1]

    # Convert to numpy and process each sample
    gravity_output = []
    for b in range(batch_size):
        # z_sample = depth_map[b, :, :, 0].numpy()
        z_sample = depth_map[b, :, :, 0]
        g = FW_Granser(z_sample*10, r0, lambda_, nx, ny, dx, dy, n, const)
        gravity_output.append(g)

    # Convert back to tensor and add channel dimension
    gravity_field = tf.convert_to_tensor(np.array(gravity_output), dtype=tf.float32)
    return tf.expand_dims(gravity_field, -1)#*10**(-5)  # Convert to mGal
