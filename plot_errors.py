import numpy as np
from regression import Regression
from timeseries_regression import TimeseriesRegression
from trajectories.Trajectory import Trajectory, Pattern


def main(directory, drifters):

    ur_file, vr_file, ku_file, kv_file = None, None, None, None

    try:
        ur_file = directory+str(drifters)+"_drifters_ur.csv"
        vr_file = directory + str(drifters) + "_drifters_vr.csv"
        ku_file = directory+str(drifters)+"_drifters_ku.csv"
        kv_file = directory + str(drifters) + "_drifters_kv.csv"
    except Exception as e:
        print("File not found")

    ur = np.loadtxt(ur_file, delimiter=",") if ur_file is not None else None
    vr = np.loadtxt(vr_file, delimiter=",") if vr_file is not None else None

    #ku = np.loadtxt(ku_file, delimiter=",") if ku_file is not None else None
    #kv = np.loadtxt(kv_file, delimiter=",") if kv_file is not None else None

    reg = TimeseriesRegression()

    trajectory = Trajectory(nsamples=drifters, integration_time=3600, n_timesteps=48, pattern=Pattern.grid)
    reg.initialize_samples(trajectory=trajectory)

    reg.ur = ur
    reg.vr = vr
    #reg.ku = ku
    #reg.kv = kv

    reg.plot_errors()


if __name__ == "__main__":

    main("/Users/joshua/Desktop/gpr-drifters/model_output/", 300)

    # main(
    #     ur_file="/Users/joshua/Desktop/gpr-drifters/ur_save.csv",
    #     vr_file="/Users/joshua/Desktop/gpr-drifters/vr_save.csv"
    # )