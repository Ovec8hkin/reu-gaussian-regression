import numpy as np
import GPy
from regression import Regression
from velocity_fields import spiral_vf as svf
from trajectories.Trajectory import Trajectory, Pattern
from trajectories.Initializations import Initializations
from trajectories import get_velocity as vel
from kernels import CurlFreeKernel as cfk, DivFreeKernel as dfk
import os, sys
import matplotlib.pylab as pl
import time


class TimeseriesRegression(Regression):

    def __init__(self):
        super(TimeseriesRegression, self).__init__()
        self.dim = 3
        self.t = np.empty(shape=(1, 1), dtype=np.float64)
        self.T = np.empty(shape=(1, 1), dtype=np.float64)

    def create_and_shape_grid_3D(self, trajectory):
        print(self.x.size)
        self.t = np.linspace(0, trajectory.integration_time, trajectory.n_timesteps)
        self.X, self.T, self.Y = np.meshgrid(self.x, self.t, self.y)

        # X and Y are reshaped so as to be able to be read off as (y, x) coordinate pairs
        self.X = self.X.reshape([self.X.size, 1])
        self.Y = self.Y.reshape([self.Y.size, 1])
        self.T = self.T.reshape([self.T.size, 1])

        self.grid_points = np.concatenate([self.T, self.X, self.Y], axis=1)

    def initialize_samples(self, ndrifters=None, obs=None, Xo=None, trajectory=None, random=True):
        super(TimeseriesRegression, self).initialize_samples(ndrifters=ndrifters, obs=obs, Xo=Xo, trajectory=trajectory, random=random)

        if trajectory:
            times = self.trajectory.get_times()
        else:
            times = np.ones(shape=self.obs.shape)

        self.obs = np.concatenate([times[:, 0][:, None], self.obs], axis=1)
        self.Xo = np.concatenate([times[:, 0][:, None], self.Xo], axis=1)

        self.create_and_shape_grid_3D(trajectory=trajectory)

    def run_model(self, kernel=None, step=None):

        if step is not None:

            min = (step-1)*self.n_drifters
            max = min+self.n_drifters

            self.Xo = self.Xo[min:max, :]
            self.obs = self.obs[min:max, :]

            self.Xo[:, 0] = np.ones(shape=self.Xo[:, 0].shape)
            self.obs[:, 0] = np.ones(shape=self.obs[:, 0].shape)

        if kernel is None:

            print(self.dim)

            k = GPy.kern.RBF(input_dim=self.dim, ARD=True)

            self.model_u = GPy.models.GPRegression(self.Xo, self.obs[:, 2][:, None], k.copy())
            self.model_v = GPy.models.GPRegression(self.Xo, self.obs[:, 1][:, None], k.copy())

            self.model_u.optimize_restarts(num_restarts=3, verbose=True)
            self.model_v.optimize_restarts(num_restarts=3, verbose=True)

            Ur, Ku = self.model_u.predict(self.grid_points)  # Kr = posterior covariance
            Vr, Kv = self.model_v.predict(self.grid_points)

            #print(Ur)

            # Reshape the output velocity component matrices to be the same size and shape as
            # the inital matrices of x, y points
            self.ur = np.reshape(Ur, [self.t.size, self.y.size, self.x.size])
            self.vr = np.reshape(Vr, [self.t.size, self.y.size, self.x.size])

            #print(self.ur)
            #print(self.vr)

            self.ku = np.reshape(Ku, [self.t.size, self.y.size, self.x.size])
            self.kv = np.reshape(Kv, [self.t.size, self.y.size, self.x.size])

        else:
            self.format_obs()

            k = kernel

            self.model_u = GPy.models.GPRegression(self.Xo, self.obs[:, 0][:, None], k.copy())

            self.model_u.optimize_restarts(num_restarts=3, verbose=False)

            Ur, Ku = self.model_u.predict(self.grid_points)  # Kr = posterior covariance

            # Reshape the output velocity component matrices to be the same size and shape as
            # the inital matrices of x, y points
            self.ur = np.reshape(Ur[:Ur.size//2], [self.y.size, self.x.size])
            self.vr = np.reshape(Ur[Ur.size//2:], [self.y.size, self.x.size])

            self.ku = np.reshape(Ku[:Ur.size//2], [self.y.size, self.x.size])
            self.kv = np.reshape(Ku[Ur.size//2:], [self.y.size, self.x.size])

    def get_params(self):
        return self.x, self.y, self.u, self.v, self.ur, self.vr, self.Xo, self.ku, self.kv

    def save_data_to_file(self, directory):
        np.savetxt(directory+"regression_x.csv", self.x, fmt="%.6e", delimiter=',')
        np.savetxt(directory+"regression_y.csv", self.y, fmt="%.6e", delimiter=',')
        np.savetxt(directory+"regression_u.csv", self.u, fmt="%.6e", delimiter=',')
        np.savetxt(directory+"regression_v.csv", self.v, fmt="%.6e", delimiter=',')
        np.savetxt(directory+"regression_ur.csv", self.ur, fmt="%.6e", delimiter=',')
        np.savetxt(directory+"regression_vr.csv", self.vr, fmt="%.6e", delimiter=',')
        np.savetxt(directory+"regression_xo.csv", self.Xo, fmt="%.6e", delimiter=',')
        np.savetxt(directory+"regression_obs.csv", self.obs, fmt="%.6e", delimiter=',')
        np.savetxt(directory+"regression_ku.csv", self.ku, fmt="%.6e", delimiter=',')
        np.savetxt(directory+"regression_kv.csv", self.kv, fmt="%.6e", delimiter=',')

        if self.trajectory is not None:
            print("Saving Trajectory")
            self.trajectory.save_positions_to_file()

        np.savetxt(directory+"regression_model_u.npy", self.model_u.param_array, fmt="%.6e", delimiter=',')
        np.savetxt(directory+"regression_model_v.npy", self.model_v.param_array, fmt="%.6e", delimiter=',')

    def load_data_from_file(self, directory):

        self.x = np.loadtxt(directory+"regression_x.csv", delimiter=',')
        self.y = np.loadtxt(directory+"regression_y.csv", delimiter=',')
        self.u = np.loadtxt(directory+"regression_u.csv", delimiter=',')
        self.v = np.loadtxt(directory+"regression_v.csv", delimiter=',')
        self.ur = np.loadtxt(directory+"regression_ur.csv", delimiter=',')
        self.vr = np.loadtxt(directory+"regression_vr.csv", delimiter=',')
        self.Xo = np.loadtxt(directory+"regression_xo.csv", delimiter=',')
        self.obs = np.loadtxt(directory+"regression_obs.csv", delimiter=',')
        self.ku = np.loadtxt(directory+"regression_ku.csv", delimiter=',')
        self.kv = np.loadtxt(directory+"regression_kv.csv", delimiter=',')

        if os.path.exists(directory+'trajectory_data.csv'):
            print("Loading Trajectory")
            self.trajectory = Trajectory(0, 0, 0)
            self.trajectory.load_positions_from_file()

        k = GPy.kern.RBF(input_dim=2, variance=1)
        self.model_u = GPy.models.GPRegression(self.Xo, self.obs[:, 1][:, None], k.copy())
        self.model_v = GPy.models.GPRegression(self.Xo, self.obs[:, 0][:, None], k.copy())

        self.model_u.update_model(False)
        self.model_u.initialize_parameter()
        self.model_u[:] = np.loadtxt(directory+'regression_model_u.npy')
        self.model_u.update_model(True)

        self.model_v.update_model(False)
        self.model_v.initialize_parameter()
        self.model_v[:] = np.loadtxt(directory+'regression_model_v.npy')
        self.model_v.update_model(True)

    def format_obs(self):

        us = self.obs[:, 2][:, None]
        vs = self.obs[:, 1][:, None]

        self.obs = np.concatenate([us, vs], axis=0)

    def plot_quiver(self, show=True, save=False):

        n = 1

        self.ur = self.ur[n, :, :]
        self.vr = self.vr[n, :, :]
        self.ku = self.ku[n, :, :]
        self.kv = self.kv[n, :, :]

        self.Xo = self.Xo[self.n_drifters*n:self.n_drifters*n+self.n_drifters, :]

        print(self.ur.shape)
        print(self.vr.shape)
        print(self.x.shape)
        print(self.y.shape)

        fig1 = pl.figure(figsize=(self.figW, self.figH))

        # Plot the Original Velocity Field
        plot = fig1.add_subplot(1, 2, 1, aspect='equal')
        plot.quiver(self.x[::self.ds], self.y[::self.ds], self.u[::self.ds, ::self.ds], self.v[::self.ds, ::self.ds],
                    scale=self.scale)
        plot.streamplot(self.x, self.y, self.u, self.v)
        plot.plot(self.Xo[:, self.dim - 1], self.Xo[:, self.dim - 2], 'og', markersize=self.marker_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.set_title('Original Velocity Field (' + str(self.n_drifters) + ' drifters)', size=self.text_size,
                       pad=self.title_pad)
        plot.tick_params(labelsize=self.tick_label_size)

        # Plot the Velocity Field Generated by the Gaussian Process Regression
        plot = fig1.add_subplot(1, 2, 2, aspect='equal')
        plot.quiver(self.x[::self.ds], self.y[::self.ds], self.ur[::self.ds, ::self.ds], self.vr[::self.ds, ::self.ds],
                    scale=self.scale)
        plot.streamplot(self.x, self.y, self.ur, self.vr)
        plot.plot(self.Xo[:, self.dim - 1], self.Xo[:, self.dim - 2], 'og', markersize=self.marker_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.set_title('GPR Velocity Field (' + str(self.n_drifters) + ' drifters)', size=self.text_size,
                       pad=self.title_pad)
        plot.tick_params(labelsize=self.tick_label_size)

        pl.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)

        if show:
            pl.show()


if __name__ == "__main__":

    regression = TimeseriesRegression()

    div_k = dfk.DivFreeK(3)
    curl_k = cfk.CurlFreeK(3)

    kernel = div_k + curl_k

    trajectory = Trajectory(nsamples=60, integration_time=300, n_timesteps=30, pattern=Pattern.grid)
    #regression.initialize_samples(nsamples=150)
    regression.initialize_samples(trajectory=trajectory)

    np.set_printoptions(threshold=np.nan)

    print(regression.X.size)
    print(regression.Y.size)
    print(regression.T.size)
    print(regression.obs.size)
    print(regression.Xo.size)

    regression.run_model()

    #print(regression.model_u.kern.lengthscale[2])

    regression.plot_errors()
