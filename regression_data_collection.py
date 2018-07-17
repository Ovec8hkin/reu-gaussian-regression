import numpy as np

from timeseries_regression import TimeseriesRegression
from trajectories.Trajectory import Trajectory
from trajectories.Trajectory import Pattern

def main():

    drifters = np.arange(30, 330, 30)

    ux_lengthscales = []
    uy_lengthscales = []
    ut_lengthscales = []
    ugaussian_noise = []

    vx_lengthscales = []
    vy_lengthscales = []
    vt_lengthscales = []
    vgaussian_noise = []

    for d in drifters:

        print(d)

        regression = TimeseriesRegression()

        trajectory = Trajectory(nsamples=d, integration_time=3600, n_timesteps=48, pattern=Pattern.grid)

        regression.initialize_samples(d, trajectory=trajectory)
        regression.run_model()

        model_u = regression.model_u
        model_v = regression.model_v

        u_lengthscales = model_u.kern.lengthscale
        v_lengthscales = model_v.kern.lengthscale

        ux_lengthscales.append(u_lengthscales[2])
        uy_lengthscales.append(u_lengthscales[1])
        ut_lengthscales.append(u_lengthscales[0])
        ugaussian_noise.append(model_u.Gaussian_noise.variance[0])

        vx_lengthscales.append(v_lengthscales[2])
        vy_lengthscales.append(v_lengthscales[1])
        vt_lengthscales.append(v_lengthscales[0])
        vgaussian_noise.append(model_v.Gaussian_noise.variance[0])

        np.savetxt('u_x_lscale.csv', ux_lengthscales, delimiter=",")
        np.savetxt('u_y_lscale.csv', uy_lengthscales, delimiter=",")
        np.savetxt('u_t_lscale.csv', ut_lengthscales, delimiter=",")
        np.savetxt('u_gaussian_noise.csv', ugaussian_noise, delimiter=",")

        np.savetxt('v_x_lscale.csv', vx_lengthscales, delimiter=",")
        np.savetxt('v_y_lscale.csv', vy_lengthscales, delimiter=",")
        np.savetxt('v_t_lscale.csv', vt_lengthscales, delimiter=",")
        np.savetxt('v_gaussian_noise.csv', vgaussian_noise, delimiter=",")

        if d in [30, 150, 300]:

            trajectory.save_parameters_to_file(str(d)+"_trajectory_info.csv")

            file_name_base = str(d)+"_samples_"

            np.savetxt(file_name_base+"ur.csv", regression.ur, delimiter=",")
            np.savetxt(file_name_base+"vr.csv", regression.vr, delimiter=",")
            np.savetxt(file_name_base+"ku.csv", regression.ku, delimiter=",")
            np.savetxt(file_name_base+"kv.csv", regression.kv, delimiter=",")

if __name__ == "__main__":
    main()
