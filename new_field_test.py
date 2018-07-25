import matplotlib.pyplot as pl
import numpy as np
from netCDF4 import Dataset
import GPy

fo = Dataset('/Users/joshua/Desktop/gpr-drifters/raw_data/Output.nc', 'r')
x = fo.variables['lon'][:] / 1000.
y = fo.variables['lat'][:] / 1000.
u = fo.variables['u'][:]
v = fo.variables['v'][:]
t = fo.variables['time'][:]

# Select only the first 13 timesteps of drifter data
# pick one every 500 drifters

yo = y[:13, ::500]
xo = x[:13, ::500]
uo = u[:13, ::500]
vo = v[:13, ::500]

to = np.repeat(t[:13, None], np.size(xo, 1), axis=1)
to = (to - to[0]) * 24  # to is in hours, starting at 0

Xo = np.concatenate([to.reshape([-1, 1]), yo.reshape([-1, 1]), xo.reshape([-1, 1])], axis=1)
uo = uo.reshape([-1, 1])
vo = vo.reshape([-1, 1])

k = GPy.kern.Matern52(input_dim=3, ARD=True)  # the type of covariance function shouldn't make
# a dramatic difference, so you can keep using the RBF

model_v = GPy.models.GPRegression(Xo, vo, k.copy())  # +k.copy()) # I used a sum of matern52 functions
# but this is not necessary for this test.

model_u = GPy.models.GPRegression(Xo, uo, k.copy())  # +k.copy())

model_v.optimize_restarts(num_restarts=3, verbose=True)
model_u.optimize_restarts(num_restarts=3, verbose=True)

# Now, load the grid points

fg = Dataset('/Users/joshua/Desktop/gpr-drifters/raw_data/osprein.nc', 'r')
timeStep = 2
xg = fg.variables['lon'][::2] / 1000.  # I used a coarser resolution to speed up computations
yg = fg.variables['lat'][::2] / 1000.
tg = 24 * (fg.variables['time'][timeStep] - fg.variables['time'][1])

ug = fg.variables['uvel'][timeStep, 0, ::2, ::2].squeeze()
vg = fg.variables['vvel'][timeStep, 0, ::2, ::2].squeeze()
# timeStep contains the index of the timeStep you want to plot
# ug and vg are originally 4D (t,z,y,x), but there is only one z level...

Yg, Tg, Xg = np.meshgrid(yg, tg, xg)  # check the order here, this could be messing up your results;
# it has to be consistent with how you defined Xo

Tr = np.reshape(Tg, [Tg.size, 1])  # reshaping grid coordinates
Yr = np.reshape(Yg, [Yg.size, 1])
Xr = np.reshape(Xg, [Xg.size, 1])

Grid = np.concatenate([Tr, Yr, Xr], axis=1)

U, UVar = model_u.predict(Grid)
V, VVar = model_v.predict(Grid)

U = U.reshape([tg.size, yg.size, xg.size])
V = V.reshape([tg.size, yg.size, xg.size])

figW = 12
figH = 6
fig = pl.figure(figsize=(figW, figH))

ds = 8  # plot one every 8 grid points to reduce the density of vectors and improve visualization

plot1 = fig.add_subplot(121)
plot1.quiver(xg[::ds], yg[::ds], ug[::ds, ::ds], vg[::ds, ::ds], scale=20)
plot1.plot(Xo[:, 2], Xo[:, 1], '.g')
plot1.axis('equal')

# If U and V are probably 3D, with dimensions 1 x yg.size x xg.size
# use squeeze to eliminate the first dimension
U = U.squeeze()
V = V.squeeze()

plot2 = fig.add_subplot(122)
plot2.quiver(xg[::ds], yg[::ds], U[::ds, ::ds], V[::ds, ::ds], scale=20)
plot2.plot(Xo[:, 2], Xo[:, 1], '.g')
plot2.axis('equal')

pl.show()



