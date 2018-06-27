import numpy as np
import math
import random

def create_centered_discrete_grid(smin, smax, nptc):
    ds = ((smax - smin) + 1) / nptc

    print(ds)

    sa = np.empty(shape=(nptc + 1, 1))
    for i in range(nptc + 1):
        sa[i] = i * ds + smin

    return sa


def initialize_particles_grid(positions, nparticles, density=None):
    xrmin, xrmax = -8.0, 8.0

    if density is not None:
        dens = density
        dist = (math.sqrt(nparticles * 1 / dens)) / 2
        xrmin = -dist
        xrmax = dist

    n = int(math.sqrt(nparticles))
    sa = create_centered_discrete_grid(xrmin, xrmax, n)

    p = positions
    ip = 0
    for i in range(n):
        for j in range(n):
            p[ip + j][0] = sa[j]
            p[ip + j][1] = sa[i]
        ip = ip + n

    return p


def initialize_particles_random(positions, nparticles):
    x = np.arange(-5, 5.1, 0.1)
    y = np.arange(-5, 5.1, 0.1)

    vals = np.arange(-5, 5.1, 0.1)

    x_choices = np.random.randint(len(vals), size=nparticles + 1)
    y_choices = np.random.randint(len(vals), size=nparticles + 1)

    p = positions

    for i in range(1):
        for j in range(nparticles):
            p[j][0] = x[x_choices[j]]
            p[j][1] = y[y_choices[j]]

    return p