import numpy as np
from trajectories import trajectory_initalizations as init


def get_velocity(pos, time=1, ndim=2):

    X = pos[0]
    Y = pos[1]
    A = 1.5
    lx, ly = 3.0, 3.0
    ratio = 0.8

    exp = np.exp(-(X**2)/lx - (Y**2)/ly)
    dpdy = ((-2*A*Y)/ly)*exp
    dpdx = ((-2*A*X)/lx)*exp

    # non-divergent velocity components
    u1 = dpdy
    v1 = -dpdx

    # non-rotational velocity components
    u2 = dpdx
    v2 = dpdy

    u = u1*ratio + (1-ratio)*u2
    v = v1*ratio + (1-ratio)*v2

    velocity = [u,v]

    return np.array(velocity)


def rk4_step(pos, time, dt, dim):
    ua = get_velocity(pos, time, dim)

    # temp_pos = np.add(pos, 0.5*dt*ua)
    temp_pos = np.add(pos, np.multiply(ua, 0.5 * dt))
    # print("UA: "+str(temp_pos))

    ut = get_velocity(temp_pos, time + 0.5 * dt, dim)
    # temp_pos = np.add(pos, 0.5*dt*ut)
    temp_pos = np.add(pos, np.multiply(ut, 0.5 * dt))
    ua = np.add(ua, np.multiply(ut, 2.0))

    ut = get_velocity(temp_pos, time + 0.5 * dt, dim)
    # temp_pos = np.add(pos, dt*ut)
    temp_pos = np.add(pos, np.multiply(ut, dt))
    ua = np.add(ua, np.multiply(ut, 2.0))

    ut = get_velocity(temp_pos, time + dt, dim)
    # pos = np.divide(np.add(p, dt*(ua*ut)), 6.0)
    # pos = np.divide(np.add(pos, np.multiply(np.add(ua, ut), dt)), 6.0)
    pos = np.add(pos, np.multiply(dt, np.divide(np.add(ua, ut), 6.0)))

    # print("POS: "+str(pos))

    return pos


def lagtransport(n_particles, integration_time, n_timesteps, density=None):
    n_dims = 2
    isnap = 5

    positions = np.empty(shape=(n_particles, n_dims))

    dt = integration_time / n_timesteps

    temp_pos = []

    positions = init.initialize_particles_grid(positions, n_particles, density=density)
    # positions = init.initialize_particles_random(positions, n_particles)

    initial = np.copy(positions)

    for i in range(n_timesteps):
        time = i * dt
        for j in range(n_particles):
            positions[j] = rk4_step(positions[j], time, dt, n_dims)
            if i % isnap == 0:
                temp_pos.append(np.copy(positions[j]))

    tem = np.array(temp_pos)

    return initial, positions, tem


if __name__ == "__main__":

    lagtransport(30, 20, 20)