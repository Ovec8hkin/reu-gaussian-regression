import GPy
import netCDF4 as nc
import numpy as np
import matplotlib.pylab as pl
from matplotlib.widgets import Slider
from timeseries_regression import TimeseriesRegression
from velocity_fields import new_vf as vf
import os, sys

class NewRegression():

    def __init__(self):
        super(NewRegression, self).__init__()
        self.dim = 3
        self.uo = np.empty(shape=(1, 1))
        self.vo = np.empty(shape=(1, 1))

        self.ds = 8  # What is this property?
        self.figW = 11.
        self.figH = 6.

        self.marker_size = 8
        self.text_size = 18
        self.scale = 20
        self.title_pad = 15
        self.tick_label_size = 13
        self.cbar_label_size = 12

    def create_and_shape_grid_3D(self, step):

        fg = nc.Dataset('/Users/joshua/Desktop/gpr-drifters/raw_data/osprein.nc', 'r')
        self.xg = fg.variables['lon'][::2] / 1000.  # I used a coarser resolution to speed up computations
        self.yg = fg.variables['lat'][::2] / 1000.
        self.tg = 24 * (fg.variables['time'][step] - fg.variables['time'][1])

        self.ug = fg.variables['uvel'][step, 0, ::2, ::2].squeeze()
        self.vg = fg.variables['vvel'][step, 0, ::2, ::2].squeeze()
        # timeStep contains the index of the timeStep you want to plot
        # ug and vg are originally 4D (t,z,y,x), but there is only one z level...

        self.Yg, self.Tg, self.Xg = np.meshgrid(self.yg, self.tg, self.xg)  # check the order here, this could be messing up your results;
        # it has to be consistent with how you defined Xo

        Tr = np.reshape(self.Tg, [self.Tg.size, 1])  # reshaping grid coordinates
        Yr = np.reshape(self.Yg, [self.Yg.size, 1])
        Xr = np.reshape(self.Xg, [self.Xg.size, 1])

        self.grid = np.concatenate([Tr, Yr, Xr], axis=1)

    def initialize_samples_from_file(self, ndrifters=1):

        fo = nc.Dataset('/Users/joshua/Desktop/gpr-drifters/raw_data/Output-new.nc', 'r')
        x = fo.variables['lon'][:] / 1000.
        y = fo.variables['lat'][:] / 1000.
        u = fo.variables['u'][:]
        v = fo.variables['v'][:]
        t = fo.variables['time'][:]

        #ii = np.where((x[0, :] >= -60) & (x[0, :] <= 80) & (y[0, :] >= -80) & (y[0, :] <= 30))[0]

        self.ndrifters = ndrifters

        print(x.size)

        ndrifters *= 101.533333

        delta = int(x.size//ndrifters)

        print(delta)

        self.yo = y[:3, ::delta]
        self.xo = x[:3, ::delta]
        self.uo = u[:3, ::delta]
        self.vo = v[:3, ::delta]

        # ii = ii[::200]
        #
        # print(ii)
        # print(ii.size)
        #
        # self.yo = y[:3, ii]
        # self.xo = x[:3, ii]
        # self.uo = u[:3, ii]
        # self.vo = v[:3, ii]

        self.to = np.repeat(t[:3, None], np.size(self.xo, 1), axis=1)
        self.to = (self.to - self.to[0]) * 24  # to is in hours, starting at 0

        self.Xo = np.concatenate([self.to.reshape([-1, 1]), self.yo.reshape([-1, 1]), self.xo.reshape([-1, 1])], axis=1)
        self.uo = self.uo.reshape([-1, 1])
        self.vo = self.vo.reshape([-1, 1])

    def run_model(self, kernel=None, step=None):

        self.create_and_shape_grid_3D(step=2)

        k = GPy.kern.RBF(input_dim=self.dim, ARD=True)

        self.model_v = GPy.models.GPRegression(self.Xo, self.vo, k.copy())
        self.model_u = GPy.models.GPRegression(self.Xo, self.uo, k.copy())

        self.model_v.optimize_restarts(num_restarts=3, verbose=True)
        self.model_u.optimize_restarts(num_restarts=3, verbose=True)

        self.U, self.UVar = self.model_u.predict(self.grid)
        self.V, self.VVar = self.model_v.predict(self.grid)

        self.U = self.U.reshape([self.tg.size, self.yg.size, self.xg.size])
        self.V = self.V.reshape([self.tg.size, self.yg.size, self.xg.size])

        # self.ur = np.reshape(Ur, [self.t.size, self.y.size, self.x.size])
        # self.vr = np.reshape(Vr, [self.t.size, self.y.size, self.x.size])
        #
        # self.ku = np.reshape(Ku, [self.t.size, self.y.size, self.x.size])
        # self.kv = np.reshape(Kv, [self.t.size, self.y.size, self.x.size])


    def plot_quiver(self, show=True, save=False):

        fig = pl.figure(figsize=(self.figW, self.figH))

        plot_extent = [self.xg.min(), self.xg.max(), self.yg.min(), self.yg.max()]
        xmin = int(np.nanmin(self.xg, axis=0))
        xmax = int(np.nanmax(self.xg, axis=0))
        ymin = int(np.nanmin(self.yg, axis=0))
        ymax = int(np.nanmax(self.yg, axis=0))

        plot1 = fig.add_subplot(111)
        plot1.quiver(self.xg[::self.ds], self.yg[::self.ds],
                     self.ug[::self.ds, ::self.ds], self.vg[::self.ds, ::self.ds],
                     scale=20)
        im = plot1.imshow(np.sqrt(self.ug**2 + self.vg**2), vmin=0, vmax=2, origin='center', extent=plot_extent, cmap='Reds')
        #plot1.plot(self.Xo[:, 2], self.Xo[:, 1], '.k', markersize=self.marker_size)
        #plot1.set_title("Initial Velocity Field \n("+str(self.ndrifters)+" drifters)", pad=self.title_pad, size=self.text_size)
        plot1.set_xlim(xmin, xmax)
        plot1.set_ylim(ymin, ymax)
        plot1.tick_params(labelsize=self.tick_label_size)
        plot1.set_xticks([int(xmin), int(xmin/2), 0, int(xmax/2), int(xmax)])
        plot1.set_yticks([int(ymin), int(ymin/2), 0, int(ymax/2), int(ymax)])

        # If U and V are probably 3D, with dimensions 1 x yg.size x xg.size
        # use squeeze to eliminate the first dimension
        # U = self.U.squeeze()
        # V = self.V.squeeze()
        #
        # plot2 = fig.add_subplot(122)
        # plot2.quiver(self.xg[::self.ds], self.yg[::self.ds],
        #              U[::self.ds, ::self.ds], V[::self.ds, ::self.ds],
        #              scale=self.scale)
        # im = plot2.imshow(np.sqrt(U ** 2 + V ** 2), vmin=0, vmax=2, origin='center', extent=plot_extent, cmap='bwr')
        # plot2.plot(self.Xo[:, 2], self.Xo[:, 1], '.k', markersize=self.marker_size)
        # plot2.set_title("Regression Velocity Field \n("+str(self.ndrifters)+" drifters)", pad=self.title_pad, size=self.text_size)
        # plot2.set_xlim(xmin, xmax)
        # plot2.set_ylim(ymin, ymax)
        # plot2.tick_params(labelsize=self.tick_label_size)
        # plot2.set_xticks([int(xmin), int(xmin / 2), 0, int(xmax / 2), int(xmax)])
        # plot2.set_yticks([int(ymin), int(ymin / 2), 0, int(ymax / 2), int(ymax)])
        # plot2.yaxis.set_visible(False)

        cb_ax = fig.add_axes([0.73, 0.124, 0.02, 0.743])
        cbar = fig.colorbar(im, cax=cb_ax)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar.set_ticks([0.0, 1.0, 2.0])

        pl.show()

        if save:
            fig.savefig("quiver.png")

    def plot_curl(self, show=True, save=False):

        def plot_curl(plot, x, y, curl, xo, title=""):

            xmin = int(np.nanmin(x, axis=0))
            xmax = int(np.nanmax(x, axis=0))
            ymin = int(np.nanmin(y, axis=0))
            ymax = int(np.nanmax(y, axis=0))

            im = plot.imshow(curl, vmin=-2, vmax=2, origin='center', extent=plot_extent, cmap='bwr')
            plot.plot(xo[:, 2], xo[:, 1], '.k', markersize=self.marker_size/1.5)
            plot.set_xlim(xmin, xmax)
            plot.set_ylim(ymin, ymax)
            plot.set_title(title, size=self.text_size, pad=self.title_pad)
            plot.tick_params(labelsize=self.tick_label_size)
            plot.set_xticks([xmin, 0, xmax])
            plot.set_yticks([ymin, 0, ymax])
            plot.yaxis.set_visible(False)
            #plot.axis('equal')
            return im

        fig = pl.figure(figsize=(self.figW, self.figH))

        plot_extent = [self.xg.min(), self.xg.max(), self.yg.min(), self.yg.max()]

        # If U and V are probably 3D, with dimensions 1 x yg.size x xg.size
        # use squeeze to eliminate the first dimension
        U = self.U.squeeze()
        V = self.V.squeeze()

        orig_curl = vf.NewVectorField.get_curl(self.ug, self.vg)
        reg_curl = vf.NewVectorField.get_curl(U, V)
        diff_curl = orig_curl - reg_curl

        init_curl_plot = fig.add_subplot(131)
        o_title = "Initial Curl"
        o_im = plot_curl(init_curl_plot, self.xg, self.yg, orig_curl, self.Xo, title=o_title)
        init_curl_plot.yaxis.set_visible(True)

        reg_curl_plot = fig.add_subplot(132)
        r_title = "Regression Curl"
        r_im = plot_curl(reg_curl_plot, self.xg, self.yg, reg_curl, self.Xo, title=r_title)

        error_curl_plot = fig.add_subplot(133)
        e_title = "Curl Error"
        e_im = plot_curl(error_curl_plot, self.xg, self.yg, diff_curl, self.Xo, title=e_title)

        cb_ax = fig.add_axes([0.92, 0.254, 0.02, 0.493])
        cbar = fig.colorbar(e_im, cax=cb_ax)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar.set_ticks([-2.0, 0.0, 2.0])

        fig.subplots_adjust(left=0.08, bottom=0.11, right=0.9, top=0.88, wspace=0.25, hspace=0.20)

        pl.show()

        if save:
            fig.savefig("curl.png")

    def plot_div(self, show=True, save=False):

        def plot_div(plot, x, y, div, xo, title=""):

            xmin = int(np.nanmin(x, axis=0))
            xmax = int(np.nanmax(x, axis=0))
            ymin = int(np.nanmin(y, axis=0))
            ymax = int(np.nanmax(y, axis=0))

            im = plot.imshow(div, vmin=-1, vmax=1, origin='center', extent=plot_extent, cmap='bwr')
            plot.plot(xo[:, 2], xo[:, 1], '.k', markersize=self.marker_size/1.5)
            plot.set_xlim(xmin, xmax)
            plot.set_ylim(ymin, ymax)
            plot.set_title(title, size=self.text_size, pad=self.title_pad)
            plot.tick_params(labelsize=self.tick_label_size)
            plot.set_xticks([xmin, 0, xmax])
            plot.set_yticks([ymin, 0, ymax])
            plot.yaxis.set_visible(False)
            #plot.axis('equal')
            return im

        fig = pl.figure(figsize=(self.figW, self.figH))

        plot_extent = [self.xg.min(), self.xg.max(), self.yg.min(), self.yg.max()]

        # If U and V are probably 3D, with dimensions 1 x yg.size x xg.size
        # use squeeze to eliminate the first dimension
        U = self.U.squeeze()
        V = self.V.squeeze()

        orig_div = vf.NewVectorField.get_div(self.ug, self.vg)
        reg_div = vf.NewVectorField.get_div(U, V)
        diff_div = orig_div - reg_div

        init_curl_plot = fig.add_subplot(131)
        o_title = "Initial Divergence"
        o_im = plot_div(init_curl_plot, self.xg, self.yg, orig_div, self.Xo, title=o_title)
        init_curl_plot.yaxis.set_visible(True)

        reg_curl_plot = fig.add_subplot(132)
        r_title = "Regression Divergence"
        r_im = plot_div(reg_curl_plot, self.xg, self.yg, reg_div, self.Xo, title=r_title)

        error_curl_plot = fig.add_subplot(133)
        e_title = "Divergence Error"
        e_im = plot_div(error_curl_plot, self.xg, self.yg, diff_div, self.Xo, title=e_title)

        cb_ax = fig.add_axes([0.92, 0.254, 0.02, 0.493])
        cbar = fig.colorbar(e_im, cax=cb_ax)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar.set_ticks([-1.0, 0.0, 1.0])

        fig.subplots_adjust(left=0.08, bottom=0.11, right=0.9, top=0.88, wspace=0.25, hspace=0.20)

        pl.show()

        if save:
            fig.savefig("divergence.png")

    def plot_error_u(self, show=True, save=False):

        def plot_u(plot, x, y, u, xo, title=""):

            xmin = int(np.nanmin(x, axis=0))
            xmax = int(np.nanmax(x, axis=0))
            ymin = int(np.nanmin(y, axis=0))
            ymax = int(np.nanmax(y, axis=0))

            im = plot.imshow(u, vmin=-1, vmax=1, origin='center', extent=plot_extent, cmap='bwr')
            plot.plot(xo[:, 2], xo[:, 1], '.k', markersize=self.marker_size/1.5)
            plot.set_xlim(xmin, xmax)
            plot.set_ylim(ymin, ymax)
            plot.set_title(title, size=self.text_size, pad=self.title_pad)
            plot.tick_params(labelsize=self.tick_label_size)
            plot.set_xticks([xmin, 0, xmax])
            plot.set_yticks([ymin, 0, ymax])
            plot.yaxis.set_visible(False)
            #plot.axis('equal')
            return im

        fig = pl.figure(figsize=(self.figW, self.figH))

        plot_extent = [self.xg.min(), self.xg.max(), self.yg.min(), self.yg.max()]

        # If U and V are probably 3D, with dimensions 1 x yg.size x xg.size
        # use squeeze to eliminate the first dimension
        U = self.U.squeeze()
        V = self.V.squeeze()


        orig_u = self.ug
        reg_u = U
        diff_u = orig_u - reg_u

        init_curl_plot = fig.add_subplot(131)
        o_title = "Initial U"
        o_im = plot_u(init_curl_plot, self.xg, self.yg, orig_u, self.Xo, title=o_title)
        init_curl_plot.yaxis.set_visible(True)

        reg_curl_plot = fig.add_subplot(132)
        r_title = "Regression U"
        r_im = plot_u(reg_curl_plot, self.xg, self.yg, reg_u, self.Xo, title=r_title)

        error_curl_plot = fig.add_subplot(133)
        e_title = "U Error"
        e_im = plot_u(error_curl_plot, self.xg, self.yg, diff_u, self.Xo, title=e_title)

        cb_ax = fig.add_axes([0.92, 0.254, 0.02, 0.493])
        cbar = fig.colorbar(e_im, cax=cb_ax)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar.set_ticks([-1.0, 0.0, 1.0])

        fig.subplots_adjust(left=0.08, bottom=0.11, right=0.9, top=0.88, wspace=0.25, hspace=0.20)

        pl.show()

        if save:
            fig.savefig("u.png")

    def plot_error_v(self, show=True, save=False):

        def plot_v(plot, x, y, v, xo, title=""):

            xmin = int(np.nanmin(x, axis=0))
            xmax = int(np.nanmax(x, axis=0))
            ymin = int(np.nanmin(y, axis=0))
            ymax = int(np.nanmax(y, axis=0))

            im = plot.imshow(v, vmin=-1, vmax=1, origin='center', extent=plot_extent, cmap='bwr')
            plot.plot(xo[:, 2], xo[:, 1], '.k', markersize=self.marker_size/1.5)
            plot.set_xlim(xmin, xmax)
            plot.set_ylim(ymin, ymax)
            plot.set_title(title, size=self.text_size, pad=self.title_pad)
            plot.tick_params(labelsize=self.tick_label_size)
            plot.set_xticks([xmin, 0, xmax])
            plot.set_yticks([ymin, 0, ymax])
            plot.yaxis.set_visible(False)
            #plot.axis('equal')
            return im

        fig = pl.figure(figsize=(self.figW, self.figH))

        plot_extent = [self.xg.min(), self.xg.max(), self.yg.min(), self.yg.max()]

        # If U and V are probably 3D, with dimensions 1 x yg.size x xg.size
        # use squeeze to eliminate the first dimension
        U = self.U.squeeze()
        V = self.V.squeeze()


        orig_v = self.vg
        reg_v = V
        diff_v = orig_v - reg_v

        init_curl_plot = fig.add_subplot(131)
        o_title = "Initial V"
        o_im = plot_v(init_curl_plot, self.xg, self.yg, orig_v, self.Xo, title=o_title)
        init_curl_plot.yaxis.set_visible(True)

        reg_curl_plot = fig.add_subplot(132)
        r_title = "Regression V"
        r_im = plot_v(reg_curl_plot, self.xg, self.yg, reg_v, self.Xo, title=r_title)

        error_curl_plot = fig.add_subplot(133)
        e_title = "V Error"
        e_im = plot_v(error_curl_plot, self.xg, self.yg, diff_v, self.Xo, title=e_title)

        cb_ax = fig.add_axes([0.92, 0.254, 0.02, 0.493])
        cbar = fig.colorbar(e_im, cax=cb_ax)
        cbar.ax.tick_params(labelsize=self.cbar_label_size)
        cbar.set_ticks([-1.0, 0.0, 1.0])

        fig.subplots_adjust(left=0.08, bottom=0.11, right=0.9, top=0.88, wspace=0.25, hspace=0.20)

        pl.show()

        if save:
            fig.savefig("v.png")


if __name__ == "__main__":

    step = sys.argv[1:]

    regression = NewRegression()
    regression.initialize_samples_from_file(ndrifters=1)
    regression.run_model()
    regression.plot_quiver(save=True)
    #regression.plot_error_u(save=True)
    #regression.plot_error_v(save=True)

    # regression = NewRegression()
    # regression.initialize_samples_from_file(ndrifters=90)
    # regression.run_model()
    # regression.plot_error_u(save=True)
    # regression.plot_error_v(save=True)
    #
    # regression = NewRegression()
    # regression.initialize_samples_from_file(ndrifters=180)
    # regression.run_model()
    # regression.plot_error_u(save=True)
    # regression.plot_error_v(save=True)


    #regression.plot_quiver()