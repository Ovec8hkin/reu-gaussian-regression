import numpy as np
import regression as reg
from trajectories.Trajectory import Trajectory

if __name__ == "__main__":

    s = np.arange(1, 181, 20)
    d = np.arange(0.01, 2.26, 0.25)

    S, D = np.meshgrid(s, d)

    zs = np.zeros(shape=(S.size, D.size))

    for i in range(len(S)):
        for j in range(len(D)):
            si = S[i][j]
            di = D[i][j]

            regression = reg.Regression(dim=3)

            trajectory = Trajectory(nsamples=si, integration_time=30, n_timesteps=10, density=di)

            regression.initialize_samples(1, trajectory=trajectory)
            regression.run_model()

            print("{}, {}".format(si, di))

            e = regression.compute_error(errors=['ge_av_raw'])

            print(e)

            zs[i][j] = e[0]

            np.savetxt("sample_density.csv", delimiter=",")