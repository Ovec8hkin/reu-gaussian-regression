import numpy as np
import GPy
from velocity_fields import spiral_vf as svf
from regression import Regression
from trajectories.Trajectory import Trajectory, Pattern
from kernels import CurlFreeKernel as cfk, DivFreeKernel as dfk
import os, sys
import matplotlib.pylab as pl
from matplotlib.widgets import Slider


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

        def setup_plot(plot, x, y, u, v, xo, title):

            plot.quiver(x[::self.ds], y[::self.ds], u[::self.ds, ::self.ds], v[::self.ds, ::self.ds], scale=self.scale)
            plot.streamplot(x, y, u, v)
            plot.plot(xo[:, self.dim - 1], xo[:, self.dim - 2], 'og', markersize=self.marker_size)
            plot.set_xlim(-5, 5)
            plot.set_ylim(-5, 5)
            plot.set_title(title, size=self.text_size, pad=self.title_pad)
            plot.tick_params(labelsize=self.tick_label_size)

        n = 0

        ur = self.ur[n, :, :]
        vr = self.vr[n, :, :]
        xo = self.Xo[self.n_drifters*n:self.n_drifters*n+self.n_drifters, :]

        fig1 = pl.figure(figsize=(self.figW, self.figH))

        # Plot the Original Velocity Field
        plot1 = fig1.add_subplot(1, 2, 1, aspect='equal')
        orig_title = 'Original Velocity Field (' + str(self.n_drifters) + ' drifters)'
        setup_plot(plot1, self.x, self.y, self.u, self.v, self.Xo, orig_title)

        # Plot the Velocity Field Generated by the Gaussian Process Regression
        plot2 = fig1.add_subplot(1, 2, 2, aspect='equal')
        gpr_title = 'GPR Velocity Field (' + str(self.n_drifters) + ' drifters)'
        setup_plot(plot2, self.x, self.y, ur, vr, xo, gpr_title)

        pl.subplots_adjust(left=0.05, bottom=0.32, right=0.95, top=0.88, wspace=0.06, hspace=0.15)

        slider_color = 'lightgoldenrodyellow'
        slider_ax = pl.axes([0.05, 0.15, 0.85, 0.05], facecolor=slider_color)

        slider = Slider(slider_ax, 'Freq', 0, self.trajectory.n_timesteps, valinit=0, valstep=1)

        def update(val):

            n = int(slider.val)
            ur = self.ur[n, :, :]
            vr = self.vr[n, :, :]
            xo = self.Xo[self.n_drifters * n:self.n_drifters * n + self.n_drifters, :]

            plot1.cla()

            setup_plot(plot1, self.x, self.y, self.u, self.v, xo, orig_title)

            plot2.cla()

            setup_plot(plot2, self.x, self.y, ur, vr, xo, gpr_title)

        slider.on_changed(update)

        if show:
            pl.show()

    def plot_curl(self, show=True, save=False):

        def plot_curl(plot, curl, title):
            im = plot.imshow(curl, vmin=0, vmax=2, origin='center', extent=plot_extent, cmap='jet')
            plot.set_title(title, size=self.text_size, pad=self.title_pad)
            plot.set_xlim(-5, 5)
            plot.set_ylim(-5, 5)
            plot.tick_params(labelsize=self.tick_label_size)
            return im

        n = 0

        ur = self.ur[n, :, :]
        vr = self.vr[n, :, :]

        init_curl = svf.SpiralVectorField.get_curl(self.u, self.v)
        reg_curl = svf.SpiralVectorField.get_curl(ur, vr)
        c_diff = init_curl - reg_curl

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig8 = pl.figure(figsize=(self.figW, self.figH))

        orig_curl_plot = fig8.add_subplot(1, 3, 1, aspect='equal')
        orig_title = "Initial Curl"
        orig_im = plot_curl(orig_curl_plot, init_curl, orig_title)

        gpr_curl_plot = fig8.add_subplot(1, 3, 2, aspect='equal')
        gpr_title = "GPR Curl"
        gpr_im = plot_curl(gpr_curl_plot, reg_curl, gpr_title)

        error_plot = fig8.add_subplot(1, 3, 3, aspect='equal')
        error_title = "Curl Error"
        error_im = plot_curl(error_plot, c_diff, error_title)

        cbar = pl.colorbar(orig_im, fraction=0.046, pad=0.04, ax=orig_curl_plot)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar = pl.colorbar(gpr_im, fraction=0.046, pad=0.04, ax=gpr_curl_plot)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar = pl.colorbar(error_im, fraction=0.046, pad=0.04, ax=error_plot)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        slider_color = 'lightgoldenrodyellow'
        slider_ax = pl.axes([0.05, 0.15, 0.85, 0.05], facecolor=slider_color)

        slider = Slider(slider_ax, 'Freq', 0, self.trajectory.n_timesteps, valinit=0, valstep=1)

        def update(val):
            n = int(slider.val)
            ur = self.ur[n, :, :]
            vr = self.vr[n, :, :]

            init_curl = svf.SpiralVectorField.get_curl(self.u, self.v)
            reg_curl = svf.SpiralVectorField.get_curl(ur, vr)
            c_diff = init_curl - reg_curl

            orig_curl_plot.cla()
            plot_curl(orig_curl_plot, init_curl, orig_title)

            gpr_curl_plot.cla()
            plot_curl(gpr_curl_plot, reg_curl, gpr_title)

            error_plot.cla()
            plot_curl(error_plot, c_diff, error_title)

        slider.on_changed(update)

        pl.subplots_adjust(left=0.06, bottom=0.47, right=0.94, top=0.88, wspace=0.48, hspace=0.0)

        if show:
            pl.show()
            
    def plot_div(self, show=True, save=False):
        
        def plot_div(plot, div, title):
            im = plot.imshow(div, vmin=-0.5, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
            plot.set_title(title, size=self.text_size, pad=self.title_pad)
            plot.set_xlim(-5, 5)
            plot.set_ylim(-5, 5)
            plot.tick_params(labelsize=self.tick_label_size)
            return im

        n = 0

        ur = self.ur[n, :, :]
        vr = self.vr[n, :, :]

        init_div = svf.SpiralVectorField.get_divergence(self.u, self.v)
        reg_div = svf.SpiralVectorField.get_divergence(ur, vr)
        diff = init_div - reg_div

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig8 = pl.figure(figsize=(self.figW, self.figH))

        orig_div_plot = fig8.add_subplot(1, 3, 1, aspect='equal')
        orig_title = "Initial Divergence"
        orig_im = plot_div(orig_div_plot, init_div, orig_title)

        gpr_div_plot = fig8.add_subplot(1, 3, 2, aspect='equal')
        gpr_title = "GPR Divergence"
        gpr_im = plot_div(gpr_div_plot, reg_div, gpr_title)

        error_plot = fig8.add_subplot(1, 3, 3, aspect='equal')
        error_title = "Divergence Error"
        error_im = plot_div(error_plot, diff, error_title)

        cbar = pl.colorbar(orig_im, fraction=0.046, pad=0.04, ax=orig_div_plot)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar = pl.colorbar(gpr_im, fraction=0.046, pad=0.04, ax=gpr_div_plot)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar = pl.colorbar(error_im, fraction=0.046, pad=0.04, ax=error_plot)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        slider_color = 'lightgoldenrodyellow'
        slider_ax = pl.axes([0.05, 0.15, 0.85, 0.05], facecolor=slider_color)

        slider = Slider(slider_ax, 'Freq', 0, self.trajectory.n_timesteps, valinit=0, valstep=1)

        def update(val):
            n = int(slider.val)
            ur = self.ur[n, :, :]
            vr = self.vr[n, :, :]

            init_div = svf.SpiralVectorField.get_divergence(self.u, self.v)
            reg_div = svf.SpiralVectorField.get_divergence(ur, vr)
            diff = init_div - reg_div

            orig_div_plot.cla()
            plot_div(orig_div_plot, init_div, orig_title)

            gpr_div_plot.cla()
            plot_div(gpr_div_plot, reg_div, gpr_title)

            error_plot.cla()
            plot_div(error_plot, diff, error_title)

        slider.on_changed(update)

        pl.subplots_adjust(left=0.06, bottom=0.47, right=0.94, top=0.88, wspace=0.48, hspace=0.0)

        if show:
            pl.show()

    def plot_raw_error(self, show=True, save=False):

        def plot_error(plot, error, title):
            im = plot.imshow(error, vmin=0, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
            plot.set_title(title, size=self.text_size, pad=self.title_pad)
            plot.set_xlim(-5, 5)
            plot.set_ylim(-5, 5)
            plot.tick_params(labelsize=self.tick_label_size)
            return im

        n = 0

        ur = self.ur[n, :, :]
        vr = self.vr[n, :, :]

        error_u = np.absolute(np.subtract(self.u, ur))
        error_v = np.absolute(np.subtract(self.v, vr))
        ea = np.sqrt(np.divide(np.add(np.power(error_u, 2), np.power(error_v, 2)), 2))

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig8 = pl.figure(figsize=(self.figW, self.figH))

        orig_div_plot = fig8.add_subplot(1, 3, 1, aspect='equal')
        orig_title = "GPR Raw Error U"
        orig_im = plot_error(orig_div_plot, error_u, orig_title)

        gpr_div_plot = fig8.add_subplot(1, 3, 2, aspect='equal')
        gpr_title = "GPR Raw Error V"
        gpr_im = plot_error(gpr_div_plot, error_v, gpr_title)

        error_plot = fig8.add_subplot(1, 3, 3, aspect='equal')
        error_title = "GPR Raw Error U+V"
        error_im = plot_error(error_plot, ea, error_title)

        cbar = pl.colorbar(orig_im, fraction=0.046, pad=0.04, ax=orig_div_plot)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar = pl.colorbar(gpr_im, fraction=0.046, pad=0.04, ax=gpr_div_plot)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar = pl.colorbar(error_im, fraction=0.046, pad=0.04, ax=error_plot)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        slider_color = 'lightgoldenrodyellow'
        slider_ax = pl.axes([0.05, 0.15, 0.85, 0.05], facecolor=slider_color)

        slider = Slider(slider_ax, 'Freq', 0, self.trajectory.n_timesteps, valinit=0, valstep=1)

        def update(val):
            n = int(slider.val)
            ur = self.ur[n, :, :]
            vr = self.vr[n, :, :]

            error_u = np.absolute(np.subtract(self.u, ur))
            error_v = np.absolute(np.subtract(self.v, vr))
            ea = np.sqrt(np.divide(np.add(np.power(error_u, 2), np.power(error_v, 2)), 2))

            orig_div_plot.cla()
            plot_error(orig_div_plot, error_u, orig_title)

            gpr_div_plot.cla()
            plot_error(gpr_div_plot, error_v, gpr_title)

            error_plot.cla()
            plot_error(error_plot, ea, error_title)

        slider.on_changed(update)

        pl.subplots_adjust(left=0.06, bottom=0.47, right=0.94, top=0.88, wspace=0.48, hspace=0.0)

        if show:
            pl.show()

    def plot_relative_error(self, show=True, save=False):

        def plot_error(plot, error, title):
            im = plot.imshow(error, vmin=0, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
            plot.set_title(title, size=self.text_size, pad=self.title_pad)
            plot.set_xlim(-5, 5)
            plot.set_ylim(-5, 5)
            plot.tick_params(labelsize=self.tick_label_size)
            return im

        n = 0

        ur = self.ur[n, :, :]
        vr = self.vr[n, :, :]

        error_u = np.absolute((self.u - ur)/self.u)
        error_v = np.absolute((self.v - vr)/self.v)
        ea = np.sqrt(np.divide(np.add(np.power(error_u, 2), np.power(error_v, 2)), 2))

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig8 = pl.figure(figsize=(self.figW, self.figH))

        orig_div_plot = fig8.add_subplot(1, 3, 1, aspect='equal')
        orig_title = "GPR Relative Error U"
        orig_im = plot_error(orig_div_plot, error_u, orig_title)

        gpr_div_plot = fig8.add_subplot(1, 3, 2, aspect='equal')
        gpr_title = "GPR Relative Error V"
        gpr_im = plot_error(gpr_div_plot, error_v, gpr_title)

        error_plot = fig8.add_subplot(1, 3, 3, aspect='equal')
        error_title = "GPR Relative Error U+V"
        error_im = plot_error(error_plot, ea, error_title)

        cbar = pl.colorbar(orig_im, fraction=0.046, pad=0.04, ax=orig_div_plot)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar = pl.colorbar(gpr_im, fraction=0.046, pad=0.04, ax=gpr_div_plot)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar = pl.colorbar(error_im, fraction=0.046, pad=0.04, ax=error_plot)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)

        slider_color = 'lightgoldenrodyellow'
        slider_ax = pl.axes([0.05, 0.15, 0.85, 0.05], facecolor=slider_color)

        slider = Slider(slider_ax, 'Freq', 0, self.trajectory.n_timesteps, valinit=0, valstep=1)

        def update(val):
            n = int(slider.val)
            ur = self.ur[n, :, :]
            vr = self.vr[n, :, :]

            error_u = np.absolute((self.u - ur) / self.u)
            error_v = np.absolute((self.v - vr) / self.v)
            ea = np.sqrt(np.divide(np.add(np.power(error_u, 2), np.power(error_v, 2)), 2))

            orig_div_plot.cla()
            plot_error(orig_div_plot, error_u, orig_title)

            gpr_div_plot.cla()
            plot_error(gpr_div_plot, error_v, gpr_title)

            error_plot.cla()
            plot_error(error_plot, ea, error_title)

        slider.on_changed(update)

        pl.subplots_adjust(left=0.06, bottom=0.47, right=0.94, top=0.88, wspace=0.48, hspace=0.0)

        if show:
            pl.show()

    def plot_errors(self, save=False):

        self.plot_quiver(save=save)
        self.plot_curl(save=save)
        self.plot_div(save=save)
        self.plot_raw_error(save=save)
        self.plot_relative_error(save=save)



if __name__ == "__main__":

    regression = TimeseriesRegression()

    div_k = dfk.DivFreeK(3)
    curl_k = cfk.CurlFreeK(3)

    kernel = div_k + curl_k

    trajectory = Trajectory(nsamples=30, integration_time=30, n_timesteps=10, pattern=Pattern.grid)
    regression.initialize_samples(trajectory=trajectory)

    regression.run_model()

    #print(regression.model_u.kern.lengthscale[2])

    regression.plot_curl()
