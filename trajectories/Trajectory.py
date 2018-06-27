import numpy as np
import enum
from trajectories.Initializations import Initializations


class Pattern(enum.Enum):
    random = 1
    grid = 2


class Trajectory:

    def __init__(self, nsamples, integration_time, n_timesteps, pattern=Pattern.random, density=None):

        self.n_particles = nsamples
        self.integration_time = integration_time
        self.n_timesteps = n_timesteps
        self.density = density
        self.pattern = pattern

        self.initial = np.empty(shape=(1, 1))
        self.positions = np.empty(shape=(1, 1))
        self.intermediates = np.empty(shape=(1, 1))

    def rk4_step(self, pos, time, dt, dim):
        ua = self.get_velocity(pos, time, dim)

        # temp_pos = np.add(pos, 0.5*dt*ua)
        temp_pos = np.add(pos, np.multiply(ua, 0.5 * dt))
        # print("UA: "+str(temp_pos))

        ut = self.get_velocity(temp_pos, time + 0.5 * dt, dim)
        # temp_pos = np.add(pos, 0.5*dt*ut)
        temp_pos = np.add(pos, np.multiply(ut, 0.5 * dt))
        ua = np.add(ua, np.multiply(ut, 2.0))

        ut = self.get_velocity(temp_pos, time + 0.5 * dt, dim)
        # temp_pos = np.add(pos, dt*ut)
        temp_pos = np.add(pos, np.multiply(ut, dt))
        ua = np.add(ua, np.multiply(ut, 2.0))

        ut = self.get_velocity(temp_pos, time + dt, dim)
        # pos = np.divide(np.add(p, dt*(ua*ut)), 6.0)
        # pos = np.divide(np.add(pos, np.multiply(np.add(ua, ut), dt)), 6.0)
        pos = np.add(pos, np.multiply(dt, np.divide(np.add(ua, ut), 6.0)))

        # print("POS: "+str(pos))

        return pos

    def get_velocity(self, pos, time=1, ndim=2):
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

    def lagtransport(self):
        n_dims = 2
        isnap = 5

        self.positions = np.empty(shape=(self.n_particles, n_dims))

        dt = self.integration_time / self.n_timesteps

        temp_pos = []

        init = Initializations(positions=self.positions, n_particles=self.n_particles, density=self.density)

        if self.pattern is Pattern.grid:
            self.positions = init.initialize_particles_grid()
        else:
            self.positions = init.initialize_particles_random()

        self.initial = np.copy(self.positions)

        for i in range(self.n_timesteps):
            time = i * dt
            for j in range(self.n_particles):
                self.positions[j] = self.rk4_step(self.positions[j], time, dt, n_dims)
                if i % isnap == 0:
                    temp_pos.append(np.copy(self.positions[j]))

        self.intermediates = np.array(temp_pos)

    def get_intermediates(self):
        return self.intermediates

    def get_params(self):
        return self.initial, self.positions, self.intermediates





