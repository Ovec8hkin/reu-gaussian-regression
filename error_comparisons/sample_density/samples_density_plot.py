import numpy as np
import regression as reg
from trajectories.Trajectory import Trajectory
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

def import_array(file):
    return np.loadtxt(file, dtype=np.float64, delimiter=",")

def f(S, D):

    zs = np.zeros(shape=(D.size, S.size))

    print(zs)

    for i in range(len(S)):
        for j in range(len(D)):

            s = S[i][j]
            d = D[i][j]

            regression = reg.Regression(dim=3)

            trajectory = Trajectory(nsamples=s, integration_time=30, n_timesteps=10, density=d)

            regression.initialize_samples(1, trajectory=trajectory)
            regression.run_model()

            print("{}, {}".format(s, d))

            e = regression.compute_error(errors=['ge_av_raw'])

            print(e)

            zs[i][j] = e[0]

    return zs


if __name__ == "__main__":

    s = np.arange(21, 181, 20)
    d = np.arange(0.01, 2.26, 0.25)

    S, D = np.meshgrid(s, d)

    z = import_array("/Users/joshua/Desktop/sample_density.csv")[:, 1:]

    print(z)

    print(len(z))
    print(len(z[0]))
    print(len(s))
    print(len(d))

    fig = pl.figure()
    ax = pl.axes(projection='3d')
    ax.set_zlim3d(0, 0.01)
    ax.set_xlim3d(0, 200)
    ax.set_ylim3d(0, 2.5)

    ax.plot_surface(S, D, z, cmap='jet', edgecolor='none')

    pl.show()
