import numpy as np
import matplotlib.pylab as pl

def main(file, name, x):

    data = np.loadtxt(file, delimiter=",")

    fig = pl.figure(figsize=(10, 5))
    plot = fig.add_subplot(1, 1, 1, aspect='equal')
    plot.plot(x, data, 'og')
    plot.set_autoscaley_on(True)
    plot.set_aspect('auto')
    plot.set_title(name+" vs Number of Samples")
    plot.set_xlabel("Number of Samples")
    plot.set_ylabel(name)

    pl.show()

if __name__ == "__main__":

    main("/Users/joshua/Desktop/gpr-drifters/model_output/x_lscale.csv", "X Lengthscale", np.arange(30, 330, 30))
    main("/Users/joshua/Desktop/gpr-drifters/model_output/y_lscale.csv", "Y Lengthscale", np.arange(30, 330, 30))
    main("/Users/joshua/Desktop/gpr-drifters/model_output/t_lscale.csv", "T Lengthscale", np.arange(30, 330, 30))
    main("/Users/joshua/Desktop/gpr-drifters/model_output/gaussian_noise.csv", "Gaussian Noise", np.arange(30, 330, 30))