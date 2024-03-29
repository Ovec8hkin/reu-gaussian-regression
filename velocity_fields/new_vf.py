import netCDF4 as ncdf
import matplotlib.pylab as pl
from matplotlib.widgets import Slider
import numpy as np


class NewVectorField:

    @classmethod
    def generate_field(cls):

        file_name = "/Users/joshua/Desktop/gpr-drifters/raw_data/osprein.nc"
        file = ncdf.Dataset(file_name, 'r')

        x = file.variables['lon'][:] / 1000.
        y = file.variables['lat'][:] / 1000.
        u = file.variables['uvel'][:]
        v = file.variables['vvel'][:]
        t = file.variables['time'][:]

        return x, y, u, v, t

    @classmethod
    def get_curl(cls, u, v):

        dudy, dudx = np.gradient(u, 0.1, axis=[0, 1])
        dvdy, dvdx = np.gradient(v, 0.1, axis=[0, 1])

        return dvdx - dudy

    @classmethod
    def get_div(cls, u, v):
        dudy, dudx = np.gradient(u, 0.1, axis=[0, 1])
        dvdy, dvdx = np.gradient(v, 0.1, axis=[0, 1])

        return dudx + dvdy


if __name__ == "__main__":

    figW = 10
    figH = 5
    ds = 20
    marker_size = 4
    text_size = 18
    scale = 5
    title_pad = 15
    tick_label_size = 16
    cbar_label_size = 12
    dim = 3

    x, y, u, v, t = NewVectorField.generate_field()

    print(x)

    x = x.squeeze()
    y = y.squeeze()
    u = u.squeeze()
    v = v.squeeze()
    t = t.squeeze()

    u = u[0]

    fig1 = pl.figure(figsize=(figW, figH))


    def setup_plot(plot, x, y, u, v, title):
        plot.quiver(x[::ds], y[::ds], u[::ds, ::ds], v[::ds, ::ds], scale=scale)
        plot.streamplot(x, y, u, v)


    n = 0

    #ur = u[n, :, :]
    #vr = v[n, :, :]

    fig1 = pl.figure(figsize=(figW, figH))

    # Plot the Original Velocity Field
    plot1 = fig1.add_subplot(1, 2, 1, aspect='equal')
    orig_title = 'Original Velocity Field (0 drifters)'
    setup_plot(plot1, x, y, u, v, orig_title)

    #slider_color = 'lightgoldenrodyellow'
    #slider_ax = pl.axes([0.05, 0.05, 0.85, 0.05], facecolor=slider_color)

    #slider = Slider(slider_ax, 'Freq', 0, t.size, valinit=0, valstep=1)


    def update(val):

        n = int(slider.val)
        ur = u[n, :, :]
        vr = v[n, :, :]

        plot1.cla()

        setup_plot(plot1, x, y, ur, vr, orig_title)


    #slider.on_changed(update)

    pl.show()