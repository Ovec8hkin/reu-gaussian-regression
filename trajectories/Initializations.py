import numpy as np
import math

class Initializations:

    def __init__(self, positions, n_particles, density=None):

        self.positions = positions
        self.n_particles = n_particles
        self.density = density

    def create_centered_discrete_grid(self, smin, smax, nptc):
        ds = ((smax - smin) + 1) / nptc

        print(ds)

        sa = np.empty(shape=(nptc + 1, 1))
        for i in range(nptc + 1):
            sa[i] = i * ds + smin

        return sa

    def initialize_particles_grid(self):
        xrmin, xrmax = -8.0, 8.0

        if self.density is not None:
            dens = self.density
            dist = (math.sqrt(self.n_particles * 1 / dens)) / 2
            xrmin = -dist
            xrmax = dist

        n = int(math.sqrt(self.n_particles))
        sa = self.create_centered_discrete_grid(xrmin, xrmax, n)

        p = self.positions
        ip = 0
        for i in range(n):
            for j in range(n):
                p[ip + j][0] = sa[j]
                p[ip + j][1] = sa[i]
            ip = ip + n

        return p

    def initialize_particles_random(self):
        x = np.arange(-5, 5.1, 0.1)
        y = np.arange(-5, 5.1, 0.1)

        vals = np.arange(-5, 5.1, 0.1)

        x_choices = np.random.randint(len(vals), size=self.n_particles + 1)
        y_choices = np.random.randint(len(vals), size=self.n_particles + 1)

        p = self.positions

        for j in range(self.n_particles):
            p[j][0] = x[x_choices[j]]
            p[j][1] = y[y_choices[j]]

        return p