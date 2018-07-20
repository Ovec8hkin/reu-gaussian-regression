import os, sys
import matplotlib.pylab as pl
import numpy as np
import GPy

from velocity_fields import spiral_vf as svf
from trajectories.Trajectory import Trajectory, Pattern
from trajectories.Initializations import Initializations
from trajectories import get_velocity as vel
from kernels import CurlFreeKernel as cfk, DivFreeKernel as dfk


class Regression:

    def __init__(self, dim=2):

        self.dim = dim

        self.n_drifters = 0

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

        self.model_u = None
        self.model_v = None
        self.trajectory = None

        self.ds = 2  # What is this property?
        self.figW = 11.
        self.figH = 6.

        self.marker_size = 4
        self.text_size = 18
        self.scale = 5
        self.title_pad = 15
        self.tick_label_size = 16
        self.cbar_label_size = 12

    def generate_vector_field(self):
        field = svf.SpiralVectorField().generate_spiral()
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

    def initialize_samples(self, ndrifters=None, obs=None, Xo=None, trajectory=None, random=True):

        if ndrifters is None and trajectory is None:
            sys.exit(0)

        if obs is not None and Xo is not None:
            self.obs = obs
            self.Xo = Xo
            return

        if trajectory is not None:

            self.n_drifters = trajectory.n_particles

            self.trajectory = trajectory

            self.trajectory.lagtransport()
            inter = self.trajectory.get_intermediates()

            vels = np.apply_along_axis(self.trajectory.get_velocity, 1, inter)

            gaussian_variance = np.random.normal(0, 0.01*np.nanmax(vels), size=vels.shape)
            vels = vels + gaussian_variance

            self.obs = np.concatenate([vels[:, 1][:, None], vels[:, 0][:, None]], axis=1)
            self.Xo = np.concatenate([inter[:, 1][:, None], inter[:, 0][:, None]], axis=1)

            return

        self.n_drifters = ndrifters

        if random:
            init = Initializations(np.zeros(shape=(ndrifters, self.dim)), ndrifters)
            grid_pos = init.initialize_particles_random()
        else:
            init = Initializations(np.zeros(shape=(ndrifters, self.dim)), ndrifters)
            grid_pos = init.initialize_particles_grid()

        vels = np.apply_along_axis(vel.get_velocity, 1, grid_pos[:, 0:2])

        gaussian_variance = np.random.normal(0, 0.01, size=vels.shape)
        vels = vels + gaussian_variance

        self.obs = np.concatenate([vels[:, 1][:, None], vels[:, 0][:, None]], axis=1)
        self.Xo = np.concatenate([grid_pos[:, 1][:, None], grid_pos[:, 0][:, None]], axis=1)

    def run_model(self, kernel=None):

        self.generate_vector_field()
        self.create_and_shape_grid()

        if kernel is None:

            k = GPy.kern.RBF(input_dim=self.dim, ARD=True)

            self.model_u = GPy.models.GPRegression(self.Xo, self.obs[:, 1][:, None], k.copy())
            self.model_v = GPy.models.GPRegression(self.Xo, self.obs[:, 0][:, None], k.copy())

            self.model_u.optimize_restarts(num_restarts=3)
            self.model_v.optimize_restarts(num_restarts=3)

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

    def plot_quiver(self, show=True, save=False):

        fig1 = pl.figure(figsize=(self.figW, self.figH))

        # Plot the Original Velocity Field
        plot = fig1.add_subplot(1, 2, 1, aspect='equal')
        plot.quiver(self.x[::self.ds], self.y[::self.ds], self.u[::self.ds, ::self.ds], self.v[::self.ds, ::self.ds], scale=self.scale)
        plot.streamplot(self.x, self.y, self.u, self.v)
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'og', markersize=self.marker_size)
        plot.set_xlim(-5, 5)
        plot.set_xlabel("km")
        plot.set_ylim(-5, 5)
        plot.set_ylabel("km")
        plot.set_title('Original Velocity Field ('+str(self.n_drifters)+' drifters)', size=self.text_size, pad=self.title_pad)
        plot.tick_params(labelsize=self.tick_label_size)

        # Plot the Velocity Field Generated by the Gaussian Process Regression
        plot = fig1.add_subplot(1, 2, 2, aspect='equal')
        plot.quiver(self.x[::self.ds], self.y[::self.ds], self.ur[::self.ds, ::self.ds], self.vr[::self.ds, ::self.ds], scale=self.scale)
        plot.streamplot(self.x, self.y, self.ur, self.vr)
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'og', markersize=self.marker_size)
        plot.set_xlim(-5, 5)
        plot.set_xlabel("km")
        plot.set_ylim(-5, 5)
        plot.set_ylabel("km")
        plot.set_title('GPR Velocity Field ('+str(self.n_drifters)+' drifters)', size=self.text_size, pad=self.title_pad)
        plot.tick_params(labelsize=self.tick_label_size)

        pl.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)

        if show:
            pl.show()

        if save:
            fig1.savefig("vel-field.png", dpi=300)

    def plot_curl(self, show=True, save=False):

        init_curl = svf.SpiralVectorField.get_curl(self.u, self.v)
        reg_curl = svf.SpiralVectorField.get_curl(self.ur, self.vr)
        c_diff = init_curl - reg_curl

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig8 = pl.figure(figsize=(self.figW, self.figH))

        plot = fig8.add_subplot(1, 3, 1, aspect='equal')
        im1 = plot.imshow(init_curl, vmin=0, vmax=2, origin='center', extent=plot_extent, cmap='jet')
        plot.set_title("Initial Curl", size=self.text_size, pad=self.title_pad)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.tick_params(labelsize=self.tick_label_size)
        cbar = fig8.colorbar(im1, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        plot = fig8.add_subplot(1, 3, 2, aspect='equal')
        im2 = plot.imshow(reg_curl, vmin=0, vmax=2, origin='center', extent=plot_extent, cmap='jet')
        plot.set_title("GPR Model Curl", size=self.text_size, pad=self.title_pad)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.tick_params(labelsize=self.tick_label_size)
        cbar = fig8.colorbar(im2, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        plot = fig8.add_subplot(1, 3, 3, aspect='equal')
        im3 = plot.imshow(c_diff, vmin=0, vmax=2, origin='center', extent=plot_extent, cmap='jet')
        plot.set_title("Curl Error", size=self.text_size, pad=self.title_pad)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.tick_params(labelsize=self.tick_label_size)
        cbar = fig8.colorbar(im3, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        pl.subplots_adjust(left=0.06, bottom=0.47, right=0.94, top=0.88, wspace=0.48, hspace=0.0)

        if show:
            pl.show()

        if save:
            fig8.savefig("curl.png", dpi=300)

    def plot_div(self, show=True, save=False):
        init_div = svf.SpiralVectorField.get_divergence(self.u, self.v)
        reg_div = svf.SpiralVectorField.get_divergence(self.ur, self.vr)
        diff = init_div - reg_div

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig8 = pl.figure(figsize=(self.figW, self.figH))

        plot = fig8.add_subplot(1, 3, 1, aspect='equal')
        im1 = plot.imshow(init_div, vmin=-0.5, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
        plot.set_title("Initial Divergence", size=self.text_size, pad=self.title_pad)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.tick_params(labelsize=self.tick_label_size)
        cbar = fig8.colorbar(im1, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        plot = fig8.add_subplot(1, 3, 2, aspect='equal')
        im2 = plot.imshow(reg_div, vmin=-0.5, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
        plot.set_title("GPR Model Divergence", size=self.text_size, pad=self.title_pad)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.tick_params(labelsize=self.tick_label_size)
        cbar = fig8.colorbar(im2, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        plot = fig8.add_subplot(1, 3, 3, aspect='equal')
        im3 = plot.imshow(diff, vmin=-0.5, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
        plot.set_title("Divergence Error", size=self.text_size, pad=self.title_pad)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.tick_params(labelsize=self.tick_label_size)
        cbar = fig8.colorbar(im3, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        pl.subplots_adjust(left=0.06, bottom=0.47, right=0.94, top=0.88, wspace=0.48, hspace=0.0)

        if show:
            pl.show()

        if save:
            fig8.savefig("divergence.png", dpi=300)

    def plot_raw_error(self, show=True, save=False):

        ue_raw, ve_raw, ea = self.compute_error(errors=["ue_raw", "ve_raw", "ea"])

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig2 = pl.figure(figsize=(self.figW, self.figH))

        plot = fig2.add_subplot(1, 3, 1, aspect='equal')
        im1 = plot.imshow(ue_raw, vmin=0, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Abs. Error U", size=self.text_size, pad=self.title_pad)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.tick_params(labelsize=self.tick_label_size)
        cbar = fig2.colorbar(im1, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        plot = fig2.add_subplot(1, 3, 2, aspect='equal')
        im2 = plot.imshow(ve_raw, vmin=0, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Abs. Error V", size=self.text_size, pad=self.title_pad)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.tick_params(labelsize=self.tick_label_size)
        cbar = fig2.colorbar(im2, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        plot = fig2.add_subplot(1, 3, 3, aspect='equal')
        im3 = plot.imshow(ea, vmin=0, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Abs. Error", size=self.text_size, pad=self.title_pad)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.tick_params(labelsize=self.tick_label_size)
        cbar = fig2.colorbar(im3, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        pl.subplots_adjust(left=0.06, bottom=0.47, right=0.94, top=0.88, wspace=0.48, hspace=0.0)

        if show:
            pl.show()

        if save:
            fig2.savefig("abs_error.png", dpi=300)

    def plot_relative_error(self, show=True, save=False):

        ue_scaled, ve_scaled, er = self.compute_error(errors=["ue_scaled", "ve_scaled", "er"])

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig3 = pl.figure(figsize=(self.figW, self.figH))

        plot = fig3.add_subplot(1, 3, 1, aspect='equal')
        im3 = plot.imshow(ue_scaled, vmin=0, vmax=10, origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Rel. Error U", size=self.text_size, pad=self.title_pad)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.tick_params(labelsize=self.tick_label_size)
        cbar = fig3.colorbar(im3, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        plot = fig3.add_subplot(1, 3, 2, aspect='equal')
        im3 = plot.imshow(ve_scaled, vmin=0, vmax=10, origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Rel. Error U", size=self.text_size, pad=self.title_pad)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.tick_params(labelsize=self.tick_label_size)
        cbar = fig3.colorbar(im3, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        plot = fig3.add_subplot(1, 3, 3, aspect='equal')
        im3 = plot.imshow(er, vmin=0, vmax=10, origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Rel. Error U", size=self.text_size, pad=self.title_pad)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        plot.tick_params(labelsize=self.tick_label_size)
        cbar = fig3.colorbar(im3, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        pl.subplots_adjust(left=0.06, bottom=0.47, right=0.94, top=0.88, wspace=0.48, hspace=0.0)

        if show:
            pl.show()

        if save:
            fig3.savefig("rel_error.png", dpi=300)

    def plot_kukv(self, show=True, save=False):

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
        fig3.colorbar(im6, fraction=0.046, pad=0.04)

        plot = fig3.add_subplot(1, 3, 3, aspect='equal')
        im7 = plot.imshow(np.sqrt(np.divide(np.add(np.power(raw_ku, 2), np.power(raw_kv, 2)), 2)), vmin=0, vmax=0.5,
                          origin='center', extent=plot_extent, cmap='jet')
        plot.plot(self.Xo[:, self.dim-1], self.Xo[:, self.dim-2], 'or', markersize=self.marker_size)
        plot.set_title("GPR Ku (Standard Deviation)", size=self.text_size)
        plot.set_xlim(-5, 5)
        plot.set_ylim(-5, 5)
        fig3.colorbar(im7, fraction=0.046, pad=0.04)

        pl.subplots_adjust(left=0.05, bottom=0.47, right=0.95, top=0.88, wspace=0.35, hspace=0.0)

        if show:
            pl.show()

        if save:
            fig3.savefig("kukv.png", dpi=300)

    def plot_errors(self, save=False):

        self.plot_quiver(show=False, save=save)

        self.plot_curl(show=False, save=save)

        self.plot_div(show=False, save=save)

        self.plot_raw_error(show=False, save=save)

        self.plot_relative_error(show=False, save=save)

        self.plot_kukv(show=False, save=save)

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

    div_k = dfk.DivFreeK(3)
    curl_k = cfk.CurlFreeK(3)
    kernel = div_k + curl_k

    # REGULAR REGRESSION TESTS

    regression = Regression()

    pattern = Pattern.grid
    trajectory = Trajectory(nsamples=50, integration_time=30, n_timesteps=15, pattern=pattern)
    regression.initialize_samples(ndrifters=10, trajectory=trajectory)
    #regression.initialize_samples(nsamples=50, random=False)
    regression.run_model()

    # TIMESERIES REGRESSION TESTS

    # regression = TimeseriesRegression()
    #
    # print(time.time())
    # trajectory = Trajectory(nsamples=60, integration_time=30, n_timesteps=15, pattern=pattern)
    # print(time.time())
    # regression.initialize_samples(nsamples=150, trajectory=trajectory)
    # #regression.initialize_samples(nsamples=150)
    # print(time.time())
    # regression.run_model()
    # print(time.time())
    #
    regression.plot_errors()
