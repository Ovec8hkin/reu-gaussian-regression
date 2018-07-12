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

    s = np.arange(1, 181, 20)
    d = np.arange(0.01, 2.26, 0.25)

    S, D = np.meshgrid(s, d)

    z = import_array("/Users/joshua/Desktop/sample_density.csv")

    print(z)

    fig = pl.figure()
    ax = pl.axes(projection='3d')

    ax.plot_surface(S, D, z, rstride=1, cstride=1,
                 cmap='viridis', edgecolor='none')

    pl.show()
