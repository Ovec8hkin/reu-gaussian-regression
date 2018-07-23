import GPy
import netCDF4 as nc
import numpy as np
import matplotlib.pylab as pl
from matplotlib.widgets import Slider
from timeseries_regression import TimeseriesRegression
from velocity_fields import new_vf as vf
import os, sys

class NewRegression(TimeseriesRegression):

    def __init__(self):
        super(NewRegression, self).__init__()

        self.uo = np.empty(shape=(1, 1))
        self.vo = np.empty(shape=(1, 1))

        self.ds = 20

        self.generate_vector_field()

    def generate_vector_field(self):

        field = vf.NewVectorField.generate_field()

        self.x = field[0]
        self.y = field[1]
        self.u = field[2]
        self.v = field[3]
        self.t = field[4]

    def create_and_shape_grid_3D(self, step):

        self.Y, self.T, self.X = np.meshgrid(self.y, self.t[step], self.x)

        # X and Y are reshaped so as to be able to be read off as (y, x) coordinate pairs
        self.X = self.X.reshape([self.X.size, 1])
        self.Y = self.Y.reshape([self.Y.size, 1])
        self.T = self.T.reshape([self.T.size, 1])

        self.grid_points = np.concatenate([self.T, self.Y, self.X], axis=1)

    def initialize_samples_from_file(self, ndrifters):

        self.n_drifters = ndrifters

        file = nc.Dataset("/Users/joshua/Desktop/gpr-drifters/raw_data/Output.nc", 'r')

        u = file.variables['u'][:]
        v = file.variables['v'][:]
        x = file.variables['lon'][:] / 1000
        y = file.variables['lat'][:] / 1000
        t = file.variables['time'][:] * 24

        delta = np.size(u, 1)//ndrifters

        u_samp = u[:, ::delta]
        v_samp = v[:, ::delta]
        x_samp = x[:, ::delta]
        y_samp = y[:, ::delta]
        t_samp = np.repeat(t[:, None], np.size(x_samp, 1), axis=1)

        to = t_samp.reshape([-1, 1])
        yo = y_samp.reshape([-1, 1])
        xo = x_samp.reshape([-1, 1])

        self.Xo = np.concatenate([to, yo, xo], axis=1)

        self.uo = u_samp.reshape([-1, 1])
        self.vo = v_samp.reshape([-1, 1])

    def run_model(self, kernel=None, step=None):

        if not os.path.exists('model_u.pkl.zip'):
            k = GPy.kern.RBF(input_dim=self.dim, ARD=True)

            self.model_u = GPy.models.GPRegression(self.Xo, self.uo, k.copy())
            self.model_v = GPy.models.GPRegression(self.Xo, self.vo, k.copy())

            self.model_u.optimize_restarts(num_restarts=3, verbose=True)
            self.model_v.optimize_restarts(num_restarts=3, verbose=True)

            self.model_u.save_model('model_u.pkl')
            self.model_v.save_model('model_v.pkl')

        self.chunk_and_predict(step)

        # self.ur = np.reshape(Ur, [self.t.size, self.y.size, self.x.size])
        # self.vr = np.reshape(Vr, [self.t.size, self.y.size, self.x.size])
        #
        # self.ku = np.reshape(Ku, [self.t.size, self.y.size, self.x.size])
        # self.kv = np.reshape(Kv, [self.t.size, self.y.size, self.x.size])

    def chunk_and_predict(self, tstep):

        import scipy.io as sio

        self.create_and_shape_grid_3D(tstep)

        self.model_u = GPy.core.GP.load_model('model_u.pkl.zip')
        self.model_v = GPy.core.GP.load_model('model_v.pkl.zip')

        print(self.model_u)
        print(self.model_v)

        Ur, Ku = self.model_u.predict(self.grid_points)
        Vr, Kv = self.model_v.predict(self.grid_points)

        self.ur = np.reshape(Ur, [self.t[step].size, self.y.size, self.x.size])
        self.vr = np.reshape(Vr, [self.t[step].size, self.y.size, self.x.size])

        out_name = "/Users/joshua/Desktop/gpr-drifters/model_output/velocity_mat_data/velocities_"+str(tstep)+".mat"
        sio.savemat(out_name, {"ur": self.ur, "vr": self.vr})

    def chunk_grid(self, tstep):

        min = tstep*self.x.size*self.y.size
        max = (tstep+1)*self.x.size*self.y.size

        grid_points = self.grid_points[min:max]

        return grid_points

    def plot_quiver(self, show=True, save=False):

        def setup_plot(plot, x, y, u, v, xo, title):

            plot.quiver(x[::self.ds], y[:v.shape[0]:self.ds], u[::self.ds, ::self.ds], v[::self.ds, ::self.ds], scale=self.scale)
            plot.streamplot(x, y[:v.shape[0]], u, v)
            plot.plot(xo[:, self.dim - 1], xo[:, self.dim - 2], 'og', markersize=self.marker_size)
            plot.set_xlim(np.nanmin(x, axis=0), np.nanmax(x, axis=0))
            plot.set_ylim(np.nanmin(y, axis=0), np.nanmax(y, axis=0))
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
        setup_plot(plot1, self.x, self.y, self.u[n, 0, :, :], self.v[n, 0, :, :], self.Xo, orig_title)

        # Plot the Velocity Field Generated by the Gaussian Process Regression
        plot2 = fig1.add_subplot(1, 2, 2, aspect='equal')
        gpr_title = 'GPR Velocity Field (' + str(self.n_drifters) + ' drifters)'
        setup_plot(plot2, self.x, self.y, ur, vr, xo, gpr_title)

        pl.subplots_adjust(left=0.05, bottom=0.20, right=0.95, top=0.88, wspace=0.06, hspace=0.15)

        slider_color = 'lightgoldenrodyellow'
        slider_ax = pl.axes([0.05, 0.05, 0.85, 0.05], facecolor=slider_color)

        slider = Slider(slider_ax, 'Freq', 0, self.n_drifters, valinit=0, valstep=1)

        def update(val):

            n = int(slider.val)
            ur = self.ur[n, :, :]
            vr = self.vr[n, :, :]
            xo = self.Xo[self.n_drifters * n:self.n_drifters * n + self.n_drifters, :]

            plot1.cla()

            setup_plot(plot1, self.x, self.y, self.u[n, 0, :, :], self.v[n, 0, :, :], xo, orig_title)

            plot2.cla()

            setup_plot(plot2, self.x, self.y, ur, vr, xo, gpr_title)

        slider.on_changed(update)

        if show:
            pl.show()

        if save:
            slider_ax.set_axis_off()
            fig1.savefig('quiver.png', dpi=300)

    def plot_curl(self, show=True, save=False):

        def plot_curl(plot, x, y, curl, xo, title):

            xmin = int(np.nanmin(x, axis=0))
            xmax = int(np.nanmax(x, axis=0))
            ymin = int(np.nanmin(y, axis=0))
            ymax = int(np.nanmax(y, axis=0))

            im = plot.imshow(curl, vmin=0, vmax=2, origin='center', extent=plot_extent, cmap='jet')
            plot.plot(xo[:, self.dim - 1], xo[:, self.dim - 2], 'og', markersize=self.marker_size)
            plot.set_xlim(xmin, xmax)
            plot.set_ylim(ymin, ymax)
            plot.set_title(title, size=self.text_size, pad=self.title_pad)
            plot.tick_params(labelsize=self.tick_label_size)
            plot.set_xticks([xmin, 0, xmax])
            return im

        n = 0

        ur = self.ur[n, :, :]
        vr = self.vr[n, :, :]
        xo = self.Xo[self.n_drifters*n:self.n_drifters*n+self.n_drifters, :]

        print("x")
        print(self.x)

        init_curl = vf.NewVectorField.get_curl(self.u[n, 0, :, :], self.v[n, 0, :, :])
        reg_curl = vf.NewVectorField.get_curl(ur, vr)
        diff_curl = init_curl - reg_curl

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig8 = pl.figure(figsize=(self.figW, self.figH))

        orig_curl_plot = fig8.add_subplot(1, 3, 1, aspect='equal')

        orig_curl_plot.set_yticks([int(np.nanmin(self.y, axis=0)), 0, int(np.nanmax(self.y, axis=0))])
        orig_title = "Initial Curl"
        orig_im = plot_curl(orig_curl_plot, self.x, self.y, init_curl, xo, orig_title)

        gpr_curl_plot = fig8.add_subplot(1, 3, 2, aspect='equal')
        gpr_curl_plot.set_yticks([])
        gpr_title = "GPR Curl"
        gpr_im = plot_curl(gpr_curl_plot, self.x, self.y, reg_curl, xo, gpr_title)

        error_plot = fig8.add_subplot(1, 3, 3, aspect='equal')
        error_plot.set_yticks([])
        error_title = "Curl Error"
        error_im = plot_curl(error_plot, self.x, self.y, diff_curl, xo, error_title)

        cb_ax = fig8.add_axes([0.93, 0.414, 0.02, 0.493])
        cbar = fig8.colorbar(error_im, cax=cb_ax)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar.set_ticks([0.0, 1.0, 2.0])

        fig8.subplots_adjust(left=0.09, bottom=0.32, right=0.92, top=1.00, wspace=0.26, hspace=0.15)

        slider_color = 'lightgoldenrodyellow'
        slider_ax = pl.axes([0.05, 0.05, 0.85, 0.05], facecolor=slider_color)

        slider = Slider(slider_ax, 'Freq', 0, self.n_drifters, valinit=0, valstep=1)

        def update(val):

            n = int(slider.val)
            ur = self.ur[n, :, :]
            vr = self.vr[n, :, :]
            xo = self.Xo[self.n_drifters * n:self.n_drifters * n + self.n_drifters, :]

            init_curl = vf.NewVectorField.get_curl(self.u[n, :, :], self.v[n, :, :])
            reg_curl = vf.NewVectorField.get_curl(ur, vr)
            c_diff = init_curl - reg_curl

            orig_curl_plot.cla()
            plot_curl(orig_curl_plot, self.x, self.y, init_curl, xo, orig_title)

            gpr_curl_plot.cla()
            plot_curl(gpr_curl_plot, self.x, self.y, reg_curl, xo, gpr_title)

            error_plot.cla()
            plot_curl(error_plot, self.x, self.y, c_diff, xo, error_title)

        slider.on_changed(update)

        if show:
            pl.show()

        if save:
            slider_ax.set_axis_off()
            fig8.savefig("curl.png", dpi=300)
            # extent1 = orig_curl_plot.get_window_extent().transformed(fig8.dpi_scale_trans.inverted())
            # fig8.savefig('orig_curl.png', dpi=300, bbox_inches=extent1.expanded(1.32, 1.33))
            # extent2 = gpr_curl_plot.get_window_extent().transformed(fig8.dpi_scale_trans.inverted())
            # fig8.savefig('reg_curl.png', dpi=300, bbox_inches=extent2.expanded(1.32, 1.33))
            # extent3 = error_plot.get_window_extent().transformed(fig8.dpi_scale_trans.inverted())
            # fig8.savefig('error_curl.png', dpi=300, bbox_inches=extent3.expanded(1.6, 1.33))

    def plot_div(self, show=True, save=False):

        def plot_div(plot, x, y, curl, xo, title):

            xmin = int(np.nanmin(x, axis=0))
            xmax = int(np.nanmax(x, axis=0))
            ymin = int(np.nanmin(y, axis=0))
            ymax = int(np.nanmax(y, axis=0))

            im = plot.imshow(curl, vmin=-0.5, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
            plot.plot(xo[:, self.dim - 1], xo[:, self.dim - 2], 'og', markersize=self.marker_size)
            plot.set_xlim(xmin, xmax)
            plot.set_ylim(ymin, ymax)
            plot.set_title(title, size=self.text_size, pad=self.title_pad)
            plot.tick_params(labelsize=self.tick_label_size)
            plot.set_xticks([xmin, 0, xmax])
            plot.set_yticks([ymin, 0, ymax])
            return im

        n = 0

        ur = self.ur[n, :, :]
        vr = self.vr[n, :, :]
        xo = self.Xo[self.n_drifters*n:self.n_drifters*n+self.n_drifters, :]

        print("x")
        print(self.x)

        init_curl = vf.NewVectorField.get_div(self.u[n, 0, :, :], self.v[n, 0, :, :])
        reg_curl = vf.NewVectorField.get_div(ur, vr)
        diff_curl = init_curl - reg_curl

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig8 = pl.figure(figsize=(self.figW, self.figH))

        orig_curl_plot = fig8.add_subplot(1, 3, 1, aspect='equal')
        orig_title = "Initial Curl"
        orig_im = plot_div(orig_curl_plot, self.x, self.y, init_curl, xo, orig_title)

        gpr_curl_plot = fig8.add_subplot(1, 3, 2, aspect='equal')
        gpr_title = "GPR Curl"
        gpr_im = plot_div(gpr_curl_plot, self.x, self.y, reg_curl, xo, gpr_title)

        error_plot = fig8.add_subplot(1, 3, 3, aspect='equal')
        error_title = "Curl Error"
        error_im = plot_div(error_plot, self.x, self.y, diff_curl, xo, error_title)

        cb_ax = fig8.add_axes([0.93, 0.414, 0.02, 0.493])
        cbar = fig8.colorbar(error_im, cax=cb_ax)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar.set_ticks([-0.5, 0, 0.5])

        fig8.subplots_adjust(left=0.09, bottom=0.32, right=0.92, top=1.00, wspace=0.26, hspace=0.15)

        slider_color = 'lightgoldenrodyellow'
        slider_ax = pl.axes([0.05, 0.05, 0.85, 0.05], facecolor=slider_color)

        slider = Slider(slider_ax, 'Freq', 0, self.n_drifters, valinit=0, valstep=1)

        def update(val):

            n = int(slider.val)
            ur = self.ur[n, :, :]
            vr = self.vr[n, :, :]
            xo = self.Xo[self.n_drifters * n:self.n_drifters * n + self.n_drifters, :]

            init_curl = vf.NewVectorField.get_div(self.u[n, 0, :, :], self.v[n, 0, :, :])
            reg_curl = vf.NewVectorField.get_div(ur, vr)
            c_diff = init_curl - reg_curl

            orig_curl_plot.cla()
            plot_div(orig_curl_plot, self.x, self.y, init_curl, xo, orig_title)

            gpr_curl_plot.cla()
            plot_div(gpr_curl_plot, self.x, self.y, reg_curl, xo, gpr_title)

            error_plot.cla()
            plot_div(error_plot, self.x, self.y, c_diff, xo, error_title)

        slider.on_changed(update)

        if show:
            pl.show()

        if save:
            slider_ax.set_axis_off()
            fig8.savefig('divergence.png', dpi=300)

    def plot_kukv(self, show=True, save=False):

        def plot_kukv(plot, x, y, k, xo, title):

            xmin = int(np.nanmin(x, axis=0))
            xmax = int(np.nanmax(x, axis=0))
            ymin = int(np.nanmin(y, axis=0))
            ymax = int(np.nanmax(y, axis=0))

            im = plot.imshow(k, vmin=0, vmax=0.5, origin='center', extent=plot_extent, cmap='jet')
            plot.plot(xo[:, self.dim - 1], xo[:, self.dim - 2], 'og', markersize=self.marker_size)
            plot.set_xlim(xmin, xmax)
            plot.set_ylim(ymin, ymax)
            plot.set_title(title, size=self.text_size, pad=self.title_pad)
            plot.tick_params(labelsize=self.tick_label_size)
            plot.set_xticks([xmin, 0, xmax])
            plot.set_yticks([ymin, 0, ymax])
            return im

        n = 0

        ku = self.ku[n, :, :]
        kv = self.kv[n, :, :]
        xo = self.Xo[self.n_drifters*n:self.n_drifters*n+self.n_drifters, :]

        print("x")
        print(self.x)

        plot_extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

        fig8 = pl.figure(figsize=(self.figW, self.figH))

        orig_curl_plot = fig8.add_subplot(1, 3, 1, aspect='equal')
        orig_title = "Initial Curl"
        orig_im = plot_kukv(orig_curl_plot, self.x, self.y, ku, xo, orig_title)

        gpr_curl_plot = fig8.add_subplot(1, 3, 2, aspect='equal')
        gpr_title = "GPR Curl"
        gpr_im = plot_kukv(gpr_curl_plot, self.x, self.y, kv, xo, gpr_title)

        error_plot = fig8.add_subplot(1, 3, 3, aspect='equal')
        error_title = "Curl Error"
        error_im = plot_kukv(error_plot, self.x, self.y, ku+kv, xo, error_title)

        cb_ax = fig8.add_axes([0.93, 0.414, 0.02, 0.493])
        cbar = fig8.colorbar(error_im, cax=cb_ax)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar.set_ticks([0, 0.25, 0.5])

        fig8.subplots_adjust(left=0.09, bottom=0.32, right=0.92, top=1.00, wspace=0.26, hspace=0.15)

        slider_color = 'lightgoldenrodyellow'
        slider_ax = pl.axes([0.05, 0.05, 0.85, 0.05], facecolor=slider_color)

        slider = Slider(slider_ax, 'Freq', 0, self.n_drifters, valinit=0, valstep=1)

        def update(val):

            n = int(slider.val)
            ku = self.ur[n, :, :]
            kv = self.vr[n, :, :]
            xo = self.Xo[self.n_drifters * n:self.n_drifters * n + self.n_drifters, :]

            orig_curl_plot.cla()
            plot_kukv(orig_curl_plot, self.x, self.y, ku, xo, orig_title)

            gpr_curl_plot.cla()
            plot_kukv(gpr_curl_plot, self.x, self.y, kv, xo, gpr_title)

            error_plot.cla()
            plot_kukv(error_plot, self.x, self.y, ku+kv, xo, error_title)

        slider.on_changed(update)

        if show:
            pl.show()

        if save:
            slider_ax.set_axis_off()
            fig8.savefig('kukv.png', dpi=300)


if __name__ == "__main__":

    step = sys.argv[1:]

    regression = NewRegression()

    print(regression.t[0].size)

    regression.initialize_samples_from_file(5)
    regression.run_model(step=0)
    #for i in range(0, regression.t.size, 1):
        #print("Step {} of {}".format(i, regression.t.size))


    #np.save('5_samples_ur', regression.ur)
    #np.save('5_samples_vr', regression.vr)

    #regression.plot_quiver()