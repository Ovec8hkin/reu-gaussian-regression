import numpy as np


class SpiralVectorField():

    @classmethod
    def generate_spiral(cls, Lx=5, Ly=5, dx=0.1, dy=0.1, lx=3.0, ly=3.0, ratio=0.8):
        x = np.arange(-1 * Lx, Lx + dx, dx)
        y = np.arange(-1 * Ly, Ly + dy, dy)
        A = 1.5
        X, Y = np.meshgrid(x, y)
        phi = A * np.exp(-(X ** 2) / lx - (Y ** 2) / ly)
        dpdy, dpdx = np.gradient(phi, dx, axis=[0, 1])

        # non-divergent velocity components
        u1 = dpdy
        v1 = -dpdx

        # non-rotational velocity components
        u2 = dpdx
        v2 = dpdy

        u = u1 * ratio + (1 - ratio) * u2
        v = v1 * ratio + (1 - ratio) * v2

        return x, y, phi, u, v

    @classmethod
    def get_velocity(cls, pos):
        X = pos[0]
        Y = pos[1]
        A = 1.5
        lx, ly = 3.0, 3.0
        ratio = 0.8

        exp = np.exp(-(X ** 2) / lx - (Y ** 2) / ly)

        dpdy = ((-2 * A * Y) / ly) * exp
        dpdx = ((-2 * A * X) / lx) * exp

        # non-divergent velocity components
        u1 = dpdy
        v1 = -dpdx

        # non-rotational velocity components
        u2 = dpdx
        v2 = dpdy

        u = u1 * ratio + (1 - ratio) * u2
        v = v1 * ratio + (1 - ratio) * v2

        velocity = [u, v]

        return np.array(velocity)

    @classmethod
    def get_curl(cls, u, v):

        dudy, dudx = np.gradient(u, 0.1, axis=[0, 1])
        dvdy, dvdx = np.gradient(v, 0.1, axis=[0, 1])

        return dvdx - dudy

    @classmethod
    def get_divergence(cls, u, v):

        dudy, dudx = np.gradient(u, 0.1, axis=[0, 1])
        dvdy, dvdx = np.gradient(v, 0.1, axis=[0, 1])

        return dudx + dvdy

    @classmethod
    def get_curl_div(cls, u, v):

        dudy, dudx = np.gradient(u, 0.1, axis=[0, 1])
        dvdy, dvdx = np.gradient(v, 0.1, axis=[0, 1])

        curl = dvdx - dudy
        div = dudx + dvdy

        return curl, div


if __name__ == "__main__":

    _, _, _, u, v = SpiralVectorField.generate_spiral()

    curl = SpiralVectorField.get_curl(u, v)

    div = SpiralVectorField.get_divergence(u, v)

    print(np.nanmax(div))
