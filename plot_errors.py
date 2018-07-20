import numpy as np
from regression import Regression
from timeseries_regression import TimeseriesRegression
from trajectories.Trajectory import Trajectory, Pattern


def main(directory, drifters):

    ur_file, vr_file, ku_file, kv_file = None, None, None, None

    try:
        ur_file = directory+str(drifters)+"_samples_ur.csv.npy"
        vr_file = directory + str(drifters) + "_samples_vr.csv.npy"
    except Exception as e:
        print("File not found")

    ur = np.load(ur_file) if ur_file is not None else None
    vr = np.load(vr_file) if vr_file is not None else None

    reg = TimeseriesRegression()

    trajectory = Trajectory(nsamples=drifters, integration_time=270, n_timesteps=48, pattern=Pattern.grid)
    reg.initialize_samples(trajectory=trajectory)

    reg.ur = ur
    reg.vr = vr

    reg.plot_quiver()


if __name__ == "__main__":

    main("/Users/joshua/Desktop/gpr-drifters/model_output/", 60)

    # main("/Users/joshua/Desktop/gpr-drifters/model_output/spiral-vector-field-test-4/", 30)
    # main("/Users/joshua/Desktop/gpr-drifters/model_output/spiral-vector-field-test-4/", 90)
    # main("/Users/joshua/Desktop/gpr-drifters/model_output/spiral-vector-field-test-4/", 180)
    # main("/Users/joshua/Desktop/gpr-drifters/model_output/spiral-vector-field-test-4/", 240)
    # main("/Users/joshua/Desktop/gpr-drifters/model_output/spiral-vector-field-test-4/", 300)

    # main(
    #     ur_file="/Users/joshua/Desktop/gpr-drifters/ur_save.csv",
    #     vr_file="/Users/joshua/Desktop/gpr-drifters/vr_save.csv"
    # )