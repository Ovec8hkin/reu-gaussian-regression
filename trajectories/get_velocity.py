import numpy as np


def get_velocity(pos, time=1, ndim=2):
    X = pos[0]
    Y = pos[1]
    A = 1.5
    lx, ly = 3.0, 3.0
    ratio = 0.8

    exp = np.exp(-(X ** 2) / lx - (Y ** 2) / ly)
    dpdy = ((-2 * A * Y) / ly) * exp
    dpdx = ((-2 * A * X) / lx) * exp

    # non-divergent velocity components
    u1 = dpdy
    v1 = -dpdx

    # non-rotational velocity components
    u2 = dpdx
    v2 = dpdy

    u = u1 * ratio + (1 - ratio) * u2
    v = v1 * ratio + (1 - ratio) * v2

    velocity = [u, v]

    return np.array(velocity)