import numpy as np

from timeseries_regression import TimeseriesRegression as Regression
from trajectories.Trajectory import Trajectory
from trajectories.Trajectory import Pattern

def main():

    drifters = np.arange(30, 330, 30)

    x_lengthscales = []
    y_lengthscales = []
    t_lengthscales = []
    gaussian_noise = []

    for d in drifters:

        print(d)

        regression = Regression()

        trajectory = Trajectory(nsamples=d, integration_time=3600, n_timesteps=48, pattern=Pattern.grid)

        regression.initialize_samples(d, trajectory=trajectory)
        regression.run_model()

        model = regression.model_u

        lengthscales = model.kern.lengthscale

        x_lengthscales.append(lengthscales[2][0])
        y_lengthscales.append(lengthscales[1][0])
        t_lengthscales.append(lengthscales[0][0])
        gaussian_noise.append(model.Gaussian_noise.variance[0])

        np.savetxt('x_lscale.csv', x_lengthscales, delimiter=",")
        np.savetxt('y_lscale.csv', y_lengthscales, delimiter=",")
        np.savetxt('t_lscale.csv', t_lengthscales, delimiter=",")
        np.savetxt('gaussian_noise.csv', gaussian_noise, delimiter=",")

        if d in [30, 150, 300]:

            file_name_base = str(d)+"_samples_"

            np.savetxt(file_name_base+"ur.csv", regression.ur, delimiter=",")
            np.savetxt(file_name_base+"vr.csv", regression.vr, delimiter=",")

if __name__ == "__main__":
    main()