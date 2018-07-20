import numpy as np
import matplotlib.pylab as pl

def main(directory, x):

    try:
        ux_lscale = directory+"u_x_lscale.csv"
        uy_lscale = directory+"u_y_lscale.csv"
        ut_lscale = directory+"u_t_lscale.csv"
        u_noise = directory+"u_gaussian_noise.csv"
        vx_lscale = directory+"v_x_lscale.csv"
        vy_lscale = directory+"v_y_lscale.csv"
        vt_lscale = directory+"v_t_lscale.csv"
        v_noise = directory+"u_gaussian_noise.csv"
    except Exception:
        print("Not found")

    uxls = np.loadtxt(ux_lscale, delimiter=",")
    vxls = np.loadtxt(vx_lscale, delimiter=",")

    uyls = np.loadtxt(uy_lscale, delimiter=",")
    vyls = np.loadtxt(vy_lscale, delimiter=",")

    utls = np.loadtxt(ut_lscale, delimiter=",")
    vtls = np.loadtxt(vt_lscale, delimiter=",")

    unoise = np.loadtxt(u_noise, delimiter=",")
    vnoise = np.loadtxt(v_noise, delimiter=",")

    fig1 = pl.figure(figsize=(10, 5))
    plot = fig1.add_subplot(1, 1, 1, aspect='equal')
    plot.plot(x, uxls, 'og')
    plot.plot(x, vxls, 'or')
    plot.set_autoscaley_on(True)
    plot.set_aspect('auto')
    plot.set_title("X Lengthscales vs Number of Drifters")
    plot.set_xlabel("Number of Samples")
    plot.set_ylabel("X Lengthscales")

    fig2 = pl.figure(figsize=(10, 5))
    plot = fig2.add_subplot(1, 1, 1, aspect='equal')
    plot.plot(x, uyls, 'og')
    plot.plot(x, vyls, 'or')
    plot.set_autoscaley_on(True)
    plot.set_aspect('auto')
    plot.set_title("Y Lengthscales vs Number of Drifters")
    plot.set_xlabel("Number of Samples")
    plot.set_ylabel("Y Lengthscales")

    fig3 = pl.figure(figsize=(10, 5))
    plot = fig3.add_subplot(1, 1, 1, aspect='equal')
    plot.plot(x, utls, 'og')
    plot.plot(x, vtls, 'or')
    plot.set_autoscaley_on(True)
    plot.set_aspect('auto')
    plot.set_title("T Lengthscales vs Number of Drifters")
    plot.set_xlabel("Number of Samples")
    plot.set_ylabel("T Lengthscales")

    fig4 = pl.figure(figsize=(10, 5))
    plot = fig4.add_subplot(1, 1, 1, aspect='equal')
    plot.plot(x, unoise, 'og')
    plot.plot(x, vnoise, 'or')
    plot.set_autoscaley_on(True)
    plot.set_aspect('auto')
    plot.set_title("Gaussian Noise vs Number of Drifters")
    plot.set_xlabel("Number of Samples")
    plot.set_ylabel("Gaussian Noise")

    pl.show()

if __name__ == "__main__":

    main("/Users/joshua/Desktop/gpr-drifters/model_output/spiral-vector-field-test-4/", np.arange(30, 300, 30))
    #main("/Users/joshua/Desktop/gpr-drifters/model_output/spiral-vector-field-test-2/u_x_lscale.csv", "X Lengthscale", np.arange(30, 180, 30))
    #main("/Users/joshua/Desktop/gpr-drifters/model_output/spiral-vector-field-test-2/u_y_lscale.csv", "Y Lengthscale", np.arange(30, 180, 30))
    #main("/Users/joshua/Desktop/gpr-drifters/model_output/spiral-vector-field-test-2/u_t_lscale.csv", "T Lengthscale", np.arange(30, 180, 30))
    #main("/Users/joshua/Desktop/gpr-drifters/model_output/spiral-vector-field-test-2/u_gaussian_noise.csv", "Gaussian Noise", np.arange(30, 180, 30))