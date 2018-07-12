import numpy as np
#import matplotlib.pylab as pl
import regression as reg
from trajectories.Trajectory import Trajectory, Pattern
from kernels import CurlFreeKernel as cfk, DivFreeKernel as dfk


def run_models(samples):
    trajectory = Trajectory(nsamples=samples, integration_time=30, n_timesteps=15, pattern=Pattern.random)

    div_k = dfk.DivFreeK(3)
    curl_k = cfk.CurlFreeK(3)

    k = div_k + curl_k

    regression = reg.Regression(dim=3)
    regression.initialize_samples(nsamples=samples, trajectory=trajectory)

    regression.run_model(kernel=k)

    cdk_e = regression.compute_error(errors=["ge_av_raw"])

    return cdk_e

def main():
    cdk_errors = []

    sample = 1
    while sample <= 300:

        print(sample)

        try:
            cdk_e = run_models(samples=sample)

            cdk_errors.append(cdk_e[0])

            sample = sample + 20

            np.savetxt("cdf_errors.csv", cdk_errors)

        except Exception as e:
            print("An error ocurred")
            print(str(e))
            pass


    #np.savetxt("rbf_errors.csv", rbf_errors)
    #np.savetxt("cdf_errors.csv", cdk_errors)

if __name__ == "__main__":
    main()
