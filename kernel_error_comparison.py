import numpy as np
#import matplotlib.pylab as pl
import regression as reg
from trajectories.Trajectory import Trajectory, Pattern
from kernels import CurlFreeKernel as cfk, DivFreeKernel as dfk


def run_models(samples):
    trajectory = Trajectory(nsamples=samples, integration_time=30, n_timesteps=30, pattern=Pattern.grid)

    div_k = dfk.DivFreeK(3)
    curl_k = cfk.CurlFreeK(3)

    k = div_k + curl_k

    regression = reg.Regression(dim=3)
    regression.initialize_samples(nsamples=samples, trajectory=trajectory)

    regression.run_model()

    rbf_e = regression.compute_error(errors=["ge_av_raw"])

    regression.run_model(kernel=k)

    cdk_e = regression.compute_error(errors=["ge_av_raw"])

    return rbf_e, cdk_e

def main():
    rbf_errors = []
    cdk_errors = []

    sample = 20
    print("Start2")
    while sample <= 300:

        print(sample)

        try:
            rbf_e, cdk_e = run_models(samples=sample)

            rbf_errors.append(rbf_e[0])
            cdk_errors.append(cdk_e[0])

            sample = sample + 20

            np.savetxt("rbf_errors.csv", rbf_errors)
            np.savetxt("cdf_errors.csv", cdk_errors)

        except Exception as e:
            print("An error ocurred")
            print(str(e))
            pass


    #np.savetxt("rbf_errors.csv", rbf_errors)
    #np.savetxt("cdf_errors.csv", cdk_errors)

if __name__ == "__main__":
    print("Start")
    main()
