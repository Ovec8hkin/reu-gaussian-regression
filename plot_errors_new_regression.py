import GPy
import numpy as np
from new_regression import NewRegression
import scipy.io as sio

def main(directory):

    regression = NewRegression()

    regression.ds = 10

    regression.generate_vector_field()
    regression.initialize_samples_from_file(30)

    ur = np.zeros(shape=(27, regression.y.size, regression.x.size))
    vr = np.zeros(shape=(27, regression.y.size, regression.x.size))
    ku = np.zeros(shape=(27, regression.y.size, regression.x.size))
    kv = np.zeros(shape=(27, regression.y.size, regression.x.size))

    # # k = GPy.kern.RBF(input_dim=3, ARD=True)
    # # regression.model_u = GPy.models.GPRegression(regression.Xo, regression.uo, k.copy())
    # # regression.model_v = GPy.models.GPRegression(regression.Xo, regression.vo, k.copy())
    #
    # regression.model_u.load_model(directory+"model_u.pkl.zip")
    # regression.model_v.load_model(directory+"model_v.pkl.zip")
    #
    # print(regression.model_u.kern.lengthscale)
    # print(regression.model_v.kern.lengthscale)

    for i in range(27):

        file = sio.loadmat(directory+"velocities_"+str(i)+".mat")

        file_ur = file['ur'][:]
        file_vr = file['vr'][:]
        file_kv = file['kv'][:]
        file_ku = file['ku'][:]

        ur[i, :, :] = file_ur
        vr[i, :, :] = file_vr
        ku[i, :, :] = file_ku
        kv[i, :, :] = file_kv

    #print(ur)

    regression.ur = ur
    regression.vr = vr
    regression.ku = np.sqrt(ku)
    regression.kv = np.sqrt(kv)

    regression.plot_quiver()
    regression.plot_curl()
    regression.plot_div()
    regression.plot_kukv()

if __name__ == "__main__":

    main("/Users/joshua/Desktop/gpr-drifters/model_output/velocity_mat_data_tsteps-2/")