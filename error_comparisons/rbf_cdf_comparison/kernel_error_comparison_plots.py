import numpy as np
import matplotlib.pylab as pl

def import_array(file):
    return np.loadtxt(file, dtype=np.float64, delimiter=",")

def compute_regression_line(x, y):
    logx = np.log10(x)
    logy = np.log10(y)

    coefficients = np.polyfit(logx[:], logy[:], 1)
    polynomial = np.poly1d(coefficients)
    ys = polynomial(logx)

    print("EQUATION: y = {}x + {}".format(coefficients[0], coefficients[1]))

    return [logx, logy, ys]

if __name__ == "__main__":

    rbf = np.absolute(import_array("/Users/joshua/Desktop/rbf_errors.csv"))
    cdf = np.absolute(import_array("/Users/joshua/Desktop/cdf_errors.csv"))

    rbf_x = np.arange(1, 500, 20)
    cdf_x = np.arange(1, 280, 20)

    logx_r, log_rbf, ys_r = compute_regression_line(rbf_x, rbf)

    logx_c, log_cdf, ys_c = compute_regression_line(cdf_x, cdf)

    fig = pl.figure(figsize=(10, 5))
    plot = fig.add_subplot(1, 1, 1, aspect='equal')
    plot.plot(rbf_x, rbf, 'og')
    plot.plot(cdf_x, cdf, 'or')
    plot.set_autoscaley_on(True)
    plot.set_aspect('auto')
    plot.set_title("Global Average Error vs Number of Samples")
    plot.set_xlabel("Number of Samples")
    plot.set_ylabel("Global Average Error")

    fig = pl.figure(figsize=(10, 5))
    plot = fig.add_subplot(1, 1, 1, aspect='equal')
    plot.plot(logx_r, log_rbf, 'og')
    plot.plot(logx_r, ys_r)
    plot.plot(logx_c, log_cdf, 'or')
    plot.plot(logx_c, ys_c)
    plot.set_autoscaley_on(True)
    plot.set_aspect('auto')
    plot.set_title("Global Average Error vs Number of Samples")
    plot.set_xlabel("Number of Samples (10^x)")
    plot.set_ylabel("Global Average Error (10^y)")

    pl.show()