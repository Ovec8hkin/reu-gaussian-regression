import numpy as np


def generate_spiral(Lx=5, Ly=5, dx=0.1, dy=0.1, lx=3., ly=3., ratio=0.8):
    x = np.arange(-1 * Lx, Lx + dx, dx)
    y = np.arange(-1 * Ly, Ly + dy, dy)
    A = 1.5
    X, Y = np.meshgrid(x, y)
    phi = A * np.exp(-(X ** 2) / lx - (Y ** 2) / ly)
    dpdy, dpdx = np.gradient(phi, dx, axis=[0, 1])

    # non-divergent velocity components
    u1 = dpdy
    v1 = -dpdx

    # non-rotational velocity components
    u2 = dpdx
    v2 = dpdy

    u = u1 * ratio + (1 - ratio) * u2
    v = v1 * ratio + (1 - ratio) * v2

    return x, y, phi, u, v