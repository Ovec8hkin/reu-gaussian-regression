import numpy as np
import GPy
from velocity_fields import spiral_vf as svf
from trajectories.Trajectory import Trajectory, Pattern
from trajectories.Initializations import Initializations
from trajectories import get_velocity as vel
from kernels import CurlFreeKernel as cfk, DivFreeKernel as dfk
import os
import matplotlib.pylab as pl


class Regression:

    def __init__(self, dim=3):

        self.dim = dim

        self.x = np.empty(shape=(1, 1))
        self.y = np.empty(shape=(1, 1))
        self.u = np.empty(shape=(1, 1))
        self.v = np.empty(shape=(1, 1))

        self.X = np.empty(shape=(1, 1))
        self.Y = np.empty(shape=(1, 1))

        self.obs = np.empty(shape=(1, 1), dtype=np.float64)
        self.Xo = np.empty(shape=(1, 1), dtype=np.float64)

        self.grid_points = np.empty(shape=(1, 1))

        self.ur = np.empty(shape=(1, 1))
        self.vr = np.empty(shape=(1, 1))
        self.ku = np.empty(shape=(1, 1))
        self.kv = np.empty(shape=(1, 1))

        self.generate_vector_field()
        self.create_and_shape_grid()

        self.model_u = None
        self.model_v = None
        self.trajectory = None

        self.ds = 2  # What is this property?
        self.figW = 10.
        self.figH = 5.

        self.marker_size = 4
        self.text_size = 10
        self.scale = 5

    def generate_vector_field(self):
        field = svf.generate_spiral()
        self.x = field[0]
        self.y = field[1]
        self.u = field[3]
        self.v = field[4]

    def create_and_shape_grid(self):
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # X and Y are reshaped so as to be able to be read off as (y, x) coordinate pairs
        self.X = self.X.reshape([self.X.size, 1])
        self.Y = self.Y.reshape([self.Y.size, 1])
        self.grid_points = np.concatenate([self.Y, self.X], axis=1)

        if self.dim == 3:

            T = np.ones(self.Y.size)
            T = T.reshape([T.size, 1])

            # A 500 X 2 2D array contatenating (y, x) coordinate points that form a grid
            self.grid_points = np.concatenate([T, self.grid_points], axis=1)

    def initialize_samples(self, nsamples, obs=None, Xo=None, trajectory=None, random=True):

        if obs is not None and Xo is not None:
            self.obs = obs
            self.Xo = Xo
            return

        if trajectory is not None:

            self.trajectory = trajectory

            self.trajectory.lagtransport()
            inter = self.trajectory.get_intermediates()

            vels = np.apply_along_axis(self.trajectory.get_velocity, 1, inter)
            self.obs = np.concatenate([vels[:, 1][:, None], vels[:, 0][:, None]], axis=1)
            self.Xo = np.concatenate([inter[:, 1][:, None], inter[:, 0][:, None]], axis=1)

            if self.dim == 3:

                times = self.trajectory.get_times()

                print(times.size)

                self.obs = np.concatenate([times[:, 0][:, None], self.obs], axis=1)
                self.Xo = np.concatenate([times[:, 0][:, None], self.Xo], axis=1)

                np.set_printoptions(threshold=np.nan)

            return

        if random:

            init = Initializations(np.zeros(shape=(nsamples, self.dim)), nsamples)
            grid_pos = init.initialize_particles_random()

            vels = np.apply_along_axis(vel.get_velocity, 1, grid_pos[:, 0:2])
            self.obs = np.concatenate([vels[:, 1][:, None], vels[:, 0][:, None]], axis=1)
            self.Xo = np.concatenate([grid_pos[:, 1][:, None], grid_pos[:, 0][:, None]], axis=1)

        else:

            init = Initializations(np.zeros(shape=(nsamples, self.dim)), nsamples, density=0.6)
            grid_pos = init.initialize_particles_grid()

            vels = np.apply_along_axis(vel.get_velocity, 1, grid_pos[:, 0:2])
            self.obs = np.concatenate([vels[:, 1][:, None], vels[:, 0][:, None]], axis=1)
            self.Xo = np.concatenate([grid_pos[:, 1][:, None], grid_pos[:, 0][:, None]], axis=1)

        if self.dim == 3:
            times = np.ones(shape=self.obs.shape)

            self.obs = np.concatenate([times[:, 0][:, None], self.obs], axis=1)
            self.Xo = np.concatenate([times[:, 0][:, None], self.Xo], axis=1)

    def run_model(self, kernel=None):

        if kernel is None:

            #print("using rbf kernel")

            k = GPy.kern.RBF(input_dim=self.dim, ARD=True)

            #print(self.obs)
            #print(self.Xo)

            self.model_u = GPy.models.GPRegression(self.Xo, self.obs[:, self.dim - 1][:, None], k.copy())
            self.model_v = GPy.models.GPRegression(self.Xo, self.obs[:, self.dim - 2][:, None], k.copy())

            self.model_u.optimize_restarts(num_restarts=5, verbose=False)
            self.model_v.optimize_restarts(num_restarts=5, verbose=False)

            Ur, Ku = self.model_u.predict(self.grid_points)  # Kr = posterior covariance
            Vr, Kv = self.model_v.predict(self.grid_points)

            # Reshape the output velocity component matrices to be the same size and shape as
            # the inital matrices of x, y points
            self.ur = np.reshape(Ur, [self.y.size, self.x.size])
            self.vr = np.reshape(Vr, [self.y.size, self.x.size])

            self.ku = np.reshape(Ku, [self.y.size, self.x.size])
            self.kv = np.reshape(Kv, [self.y.size, self.x.size])

        else:
            self.format_obs()

            print(self.obs)
            print(self.Xo)

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

    def compute_error(self, errors=None):

        ue_raw = np.absolute(np.subtract(self.u, self.ur))
        ve_raw = np.absolute(np.subtract(self.v, self.vr))

        ea = np.sqrt(np.divide(np.add(np.power(ue_raw, 2), np.power(ve_raw, 2)), 2))

        ue_scaled = np.absolute(np.divide(ue_raw, self.u))
        ve_scaled = np.absolute(np.divide(ve_raw, self.v))

        er = np.sqrt(np.divide(np.add(np.power(ue_scaled, 2), np.power(ve_scaled, 2)), 2))

        ue_av_raw = np.sum(ue_raw) / ue_raw.size
        ve_av_raw = np.sum(ve_raw) / ve_raw.size

        ue_av_scaled = np.sum(ue_scaled) / ue_scaled.size
        ve_av_scaled = np.sum(ve_scaled) / ve_scaled.size

        ge_av_raw = np.add(ue_av_raw, ve_av_raw) / 2
        ge_av_scaled = np.add(ue_av_scaled, ve_av_scaled) / 2

        error_dict = {

            "ue_raw": ue_raw,
            "ve_raw": ve_raw,
            "ea": ea,
            "ue_scaled": ue_scaled,
            "ve_scaled": ve_scaled,
            "er": er,
            "ue_av_scaled": ue_av_scaled,
            "ve_av_scaled": ve_av_scaled,
            "ge_av_raw": ge_av_raw,
            "ge_av_scaled": ge_av_scaled

        }

        if errors is not None:

            to_return = []

            for key in errors:
                to_return.append(error_dict[key])

            return to_return

        return error_dict.values()

    def plot_quiver(self, show=True):

        fig1 = pl.figure(figsize=(self.figW, self.figH))

        # Plot the Original Velocity Field
        plot = fig1.add_subplot(1, 2, 1, aspect='equal')
        plot.quiver(self.x[::self.ds], self.y[::self.ds], self.u[::self.ds, ::self.ds], self.v[::self.ds, ::self.ds], scale=self.scale)
        plot.streamplot(self.x, self.y, self.u, self.v)
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'og', markersize=self.marker_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.set_title('Original Velocity Field (30 samples)', size=self.text_size)

        # Plot the Velocity Field Generated by the Gaussian Process Regression
        plot = fig1.add_subplot(1, 2, 2, aspect='equal')
        plot.quiver(self.x[::self.ds], self.y[::self.ds], self.ur[::self.ds, ::self.ds], self.vr[::self.ds, ::self.ds], scale=self.scale)
        plot.streamplot(self.x, self.y, self.ur, self.vr)
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'og', markersize=self.marker_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.set_title('GPR Velocity Field (30 samples)', size=self.text_size)

        if show:
            pl.show()

    def plot_raw_error(self, show=True):

        ue_raw, ve_raw, ea = self.compute_error(errors=["ue_raw", "ve_raw", "ea"])

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig2 = pl.figure(figsize=(self.figW, self.figH))

        plot = fig2.add_subplot(1, 3, 1, aspect='equal')
        im1 = plot.imshow(ue_raw, vmin=0, vmax=0.5, origin='center', extent=plot_extent,
                          cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Abs. Error U", size=self.text_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        fig2.colorbar(im1, fraction=0.046, pad=0.04)

        plot = fig2.add_subplot(1, 3, 2, aspect='equal')
        im2 = plot.imshow(ve_raw, vmin=0, vmax=0.5, origin='center', extent=plot_extent,
                          cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Abs. Error V", size=self.text_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        fig2.colorbar(im2, fraction=0.046, pad=0.04)

        plot = fig2.add_subplot(1, 3, 3, aspect='equal')
        im3 = plot.imshow(ea, vmin=0, vmax=0.5, origin='center', extent=plot_extent,
                          cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Abs. Error", size=self.text_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        fig2.colorbar(im3, fraction=0.046, pad=0.04)

        if show:
            pl.show()

    def plot_relative_error(self, show=True):

        ue_scaled, ve_scaled, er = self.compute_error(errors=["ue_scaled", "ve_scaled", "er"])

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig3 = pl.figure(figsize=(self.figW, self.figH))

        plot = fig3.add_subplot(1, 3, 1, aspect='equal')
        im3 = plot.imshow(ue_scaled, vmin=0, vmax=10, origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Rel. Error U", size=self.text_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        fig3.colorbar(im3, fraction=0.046, pad=0.04)

        plot = fig3.add_subplot(1, 3, 2, aspect='equal')
        im3 = plot.imshow(ve_scaled, vmin=0, vmax=10, origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Rel. Error U", size=self.text_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        fig3.colorbar(im3, fraction=0.046, pad=0.04)

        plot = fig3.add_subplot(1, 3, 3, aspect='equal')
        im3 = plot.imshow(er, vmin=0, vmax=10, origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Rel. Error U", size=self.text_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        fig3.colorbar(im3, fraction=0.046, pad=0.04)

        if show:
            pl.show()

    def plot_kukv(self, show=True):

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        raw_ku = self.ku
        raw_kv = self.kv

        ku = np.sqrt(self.ku)
        kv = np.sqrt(self.kv)

        fig3 = pl.figure(figsize=(self.figW, self.figH))

        plot = fig3.add_subplot(1, 3, 1, aspect='equal')
        im5 = plot.imshow(ku, vmin=0, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Ku (Standard Deviation)", size=self.text_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        fig3.colorbar(im5, fraction=0.046, pad=0.04)

        plot = fig3.add_subplot(1, 3, 2, aspect='equal')
        im6 = plot.imshow(kv, vmin=0, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Ku (Standard Deviation)", size=self.text_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        fig3.colorbar(im5, fraction=0.046, pad=0.04)

        plot = fig3.add_subplot(1, 3, 3, aspect='equal')
        im7 = plot.imshow(np.sqrt(np.divide(np.add(np.power(raw_ku, 2), np.power(raw_kv, 2)), 2)), vmin=0, vmax=0.5,
                          origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Ku (Standard Deviation)", size=self.text_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        fig3.colorbar(im5, fraction=0.046, pad=0.04)

        if show:
            pl.show()

    def plot_errors(self):

        self.plot_quiver(show=False)

        self.plot_raw_error(show=False)

        self.plot_relative_error(show=False)

        self.plot_kukv(show=False)

        pl.show()

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


if __name__ == "__main__":

    regression = Regression(dim=3)

    div_k = dfk.DivFreeK(3)
    curl_k = cfk.CurlFreeK(3)

    kernel = div_k + curl_k

    trajectory = Trajectory(nsamples=30, integration_time=30, n_timesteps=15, pattern=Pattern.random)

    regression.initialize_samples(nsamples=150, trajectory=trajectory)
    regression.run_model()

    #regression.initialize_samples(nsamples=60)
    #regression.run_model()

    print(regression.model_u.kern.lengthscale[:])

    regression.plot_errors()
