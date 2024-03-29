import numpy as np
from matplotlib import pylab as pl
from velocity_fields import spiral_vf
from regression import Regression
from matplotlib.widgets import Button
from trajectories.get_velocity import get_velocity

ds = 2  # What is this property?
figW = 10.
figH = 5.

marker_size = 4
text_size = 10
scale = 5
fig = pl.figure(figsize=(figW, figH))
new_plot = None
plot = None
reg = None
scatts = []

xs = []
ys = []

def create_base_field():
    global fig, plot

    x, y, _, u, v = spiral_vf.generate_spiral()

    # Plot the Original Velocity Field
    plot = fig.add_subplot(1, 2, 1, aspect='equal')
    plot.quiver(x[::ds], y[::ds], u[::ds, ::ds], v[::ds, ::ds], scale=scale)
    plot.streamplot(x, y, u, v)
    plot.set_xlim(-5, 5)
    plot.set_ylim(-5, 5)


def show_regression_plot(event):
    global new_plot, reg

    print("SHOW REGRESSION")

    if new_plot is not None:
        new_plot.remove()

    grid_pos = np.concatenate([xs, ys], 1)

    vels = np.apply_along_axis(get_velocity, 1, grid_pos[:, 0:2])
    obs = np.concatenate([vels[:, 1][:, None], vels[:, 0][:, None]], axis=1)
    Xo = np.concatenate([grid_pos[:, 1][:, None], grid_pos[:, 0][:, None]], axis=1)

    reg = Regression(dim=2)
    reg.initialize_samples(len(Xo), obs=obs, Xo=Xo)
    reg.run_model()

    x, y, u, v, ur, vr, Xo, _, _ = reg.get_params()

    new_plot = fig.add_subplot(1, 2, 2, aspect='equal')
    new_plot.quiver(x[::ds], y[::ds], ur[::ds, ::ds], vr[::ds, ::ds], scale=scale)
    new_plot.streamplot(x, y, ur, vr)
    new_plot.plot(Xo[:, 1], Xo[:, 0], 'or', markersize=marker_size)
    new_plot.set_xlim(-5, 5)
    new_plot.set_ylim(-5, 5)
    new_plot.set_title('GPR Velocity Field (' + str(len(xs)) + ' samples)', size=text_size)

    fig.canvas.draw()


def clear(event):
    xs.clear()
    ys.clear()

    for sca in scatts:
        sca.remove()

    scatts.clear()

    fig.canvas.draw()


def show_errors(event):
    reg.plot_errors()


def place_scatter(event):
    global scatt

    if event.inaxes != plot:
        return

    scatt = plot.scatter(event.xdata, event.ydata, s=50, c='red', marker='o')  # place new sactter point
    scatts.append(scatt)
    xs.append([event.xdata])
    ys.append([event.ydata])

    fig.canvas.draw()

if __name__ == "__main__":

    create_base_field()

    ax_button_show = fig.add_axes([0.8, 0.03, 0.18, 0.045])
    show_button = Button(ax_button_show, "Show Regression")
    show_button.on_clicked(show_regression_plot)

    ax_button_hide = fig.add_axes([0.61, 0.03, 0.18, 0.045])
    hide_button = Button(ax_button_hide, "Clear")
    hide_button.on_clicked(clear)

    ax_button_hide_1 = fig.add_axes([0.42, 0.03, 0.18, 0.045])
    hide_button_1 = Button(ax_button_hide_1, "Show Error Plots")
    hide_button_1.on_clicked(show_errors)

    scatter = fig.canvas.mpl_connect('button_press_event', place_scatter)

    pl.show()

    fig.canvas.mpl_disconnect(scatter)