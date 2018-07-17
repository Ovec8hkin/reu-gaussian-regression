import numpy as np
from regression import Regression
from timeseries_regression import TimeseriesRegression
from trajectories.Trajectory import Trajectory, Pattern


def main(ur_file, vr_file):

    ur = np.loadtxt(ur_file, delimiter=",")
    vr = np.loadtxt(vr_file, delimiter=",")

    reg = TimeseriesRegression()

    trajectory = Trajectory(nsamples=30, integration_time=360, n_timesteps=48, pattern=Pattern.grid)
    reg.initialize_samples(trajectory=trajectory)

    reg.ur = ur
    reg.vr = vr

    reg.plot_errors()


if __name__ == "__main__":

    # main(
    #     ur_file="/Users/joshua/Desktop/gpr-drifters/model_output/300_samples_ur.csv",
    #     vr_file="/Users/joshua/Desktop/gpr-drifters/model_output/300_samples_vr.csv",
    # )

    main(
        ur_file="/Users/joshua/Desktop/gpr-drifters/ur_save.csv",
        vr_file="/Users/joshua/Desktop/gpr-drifters/vr_save.csv"
    )