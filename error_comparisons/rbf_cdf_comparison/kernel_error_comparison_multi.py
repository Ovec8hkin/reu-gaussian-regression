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
    regression.initialize_samples(ndrifters=samples, trajectory=trajectory)

    regression.run_model()

    rbf_e = regression.compute_error(errors=["ge_av_raw"])

    regression.run_model(kernel=k)

    cdk_e = regression.compute_error(errors=["ge_av_raw"])

    return rbf_e, cdk_e

def main():
    rbf_errors = []
    cdk_errors = []

    bounds = np.arange(1, 300, 20)

    for sample in bounds:
        attempt = 1
        local_rbf = []
        local_cdk = []
        while attempt <= 5:

            try:
                rbf_e, cdk_e = run_models(samples=sample)

                local_rbf.append(rbf_e[0])
                local_cdk.append(cdk_e[0])

                attempt = attempt + 1

            except Exception as e:
                print("An error ocurred")
                print(str(e))
                pass

        rbf = np.mean(local_rbf)
        cdk = np.mean(local_cdk)

        rbf_errors.append(rbf)
        cdk_errors.append(cdk)

        np.savetxt("rbf_errors.csv", rbf_errors)
        np.savetxt("cdf_errors.csv", cdk_errors)

if __name__ == "__main__":
    main()
