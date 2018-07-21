import GPy
import numpy as np
from new_regression import NewRegression
import scipy.io as sio

def main(directory):

    regression = NewRegression()

    regression.generate_vector_field()
    regression.initialize_samples_from_file(15)

    ur = np.zeros(shape=(regression.y.size, regression.x.size))
    vr = np.zeros(shape=(regression.y.size, regression.x.size))

    for i in range(17):

        file = sio.loadmat(directory+"velocities_"+str(i)+".mat")

        file_ur = file['ur'][:]
        file_vr = file['vr'][:]

        print(file_ur.shape)
        print(ur[i*46: (i+1)*46].shape)

        ur[i*46: (i+1)*46] = file_ur
        vr[i * 46: (i + 1) * 46] = file_vr

    regression.ur = ur
    regression.vr = vr

    regression.plot_quiver()

if __name__ == "__main__":

    main("/Users/joshua/Desktop/gpr-drifters/model_output/velocity_mat_data/")