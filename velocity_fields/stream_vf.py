import numpy as np


def make_new_vectorfield(Lx=5,Ly=5,dx=0.1,dy=0.1, m=2.0, n=2.0):

    x = np.arange(0,Lx+dx,dx)
    y = np.arange(0,Ly+dy,dy)
    X,Y = np.meshgrid(x,y)

    a, b = 5, 5

    phi = np.sin(m * np.pi * X/a) * np.sin(n * np.pi * Y/b)

    print(phi)

    dpdy,dpdx = np.gradient(phi,dx,axis=[0, 1])

    u = dpdy
    v = -dpdx

    return x, y, phi, u, v