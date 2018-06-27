import numpy as np
import GPy
from velocity_fields import spiral_vf


def regression(nsamples=15, ratio=0.8, full=False):
    '''
        nsamples: number of observations, which are randomly picked from the velocity field.
    '''

    # Generate known velocity field:
    '''
        x --> 1D array of values from -5.0 to 5.0 with a step size of 0.1
        y --> 1D array of values from -5.0 to 5.0 with a step size of 0.1

        Note: when combined, x and y form a grid with grid size of 0.1

        u --> 1D array of the x-components of the vectors at each x point
        v --> 1D array of the y-components of the vectors at each y point

    '''
    x, y, phi, u, v = spiral_vf.generate_spiral(ratio=ratio)
    #x, y, phi, u, v = stream_vf.make_new_vectorfield()

    # Generate grid from known velocity field
    '''
        X --> 	A 2D array of values from -5.0 to 5.0 with a step size of 0.1
                where each array represents a horizontal grid line of values

        Y -->	A 2D array of values from -5.0 to 5.0 with a step size of 0.1
                where each array represents a vertical grid line of values

        Note: when combined, every unique point with size of 0.1 can be read off
        from the arrays of X and Y
    '''
    X, Y = np.meshgrid(x, y)

    # X and Y are reshaped so as to be able to be read off as (y, x) coordinate pairs
    X = X.reshape([X.size, 1])
    Y = Y.reshape([Y.size, 1])

    # A 500 X 2 2D array contatenating (y, x) coordinate points that form a grid
    GridPoints = np.concatenate([Y, X], axis=1)

    U = u.reshape([u.size, 1])
    V = v.reshape([v.size, 1])

    # Take n random samples from the known velocity field
    ii = np.random.randint(0, u.size, nsamples)

    '''
        How to read the following syntax: [V[ii,0][:,None],U[ii,0][:,None]]

        Preliminary Notes:
            V = 2D array, where each of the inner arrays contains a single element
            U = 2D array, where each of the inner arrays contains a single element

            ii = 1D array of 15 elements

        V[ii, 0] --> 	An array of the elements of the first column of the inner arrays at each position
                        in the array `ii`. Note that because each inner array has only a single element, the
                        0 sepcifier is not needed, but it is a good generalization

        V[ii, 0][:, None] -->	Transforms the 1D array V[ii, 0] containing elements of the first column of
                                each inner array, into a 2D array, where each inner array contains only a 
                                single value, the value of V[ii].

    '''

    # Creates a `nsamples x 2` array where the first column consists of y components at each point ii of V, and
    # the second column consists of x components at each point ii of U.
    obs = np.concatenate([V[ii, 0][:, None], U[ii, 0][:, None]], axis=1)

    # Creates a `nsamples X 2` array where the first colum consists of the y values at each point ii of Y, and
    # the second columb consists of x components at each point ii of X.
    Xo = np.concatenate([Y[ii, 0][:, None], X[ii, 0][:, None]], axis=1)

    # Set covariance function for GPR to be RBF
    k = GPy.kern.RBF(input_dim=2, variance=1)
    # k = GPy.kern.Poly(input_dim=2, order=2, variance=1)
    # k = GPy.kern.Linear(input_dim=2, variances=1.0)
    # k = GPy.kern.MLP(input_dim=2, variance=0.5)

    # Create model for velocity components
    '''
        We are providing the regression with three pieces of data, described below:
        Xo -->      a list of (y, x) test points to construct our model with
        obs -->     a list ot (v, u) velocity components to construct our model with
        k -->       a kernel function to use

        `obs[:,1][:,None]` using the second column of `obs` we are construcitng a model
        of the x components of the velocity vectors

        This same scenario is replicated below when we create a regression model of the y 
        components of the velocity vectors.

    '''
    model_u = GPy.models.GPRegression(Xo, obs[:, 1][:, None], k.copy())  # How does this work? What is the output?
    model_v = GPy.models.GPRegression(Xo, obs[:, 0][:, None], k.copy())

    # Optimize hyper-parameters of model
    model_u.optimize_restarts(num_restarts=5, verbose=False)
    model_v.optimize_restarts(num_restarts=5, verbose=False)
    # model_u.optimize()
    # model_v.optimize()

    # Perform the regression prediciton on each point in the grid
    '''
        Having created a model from a set of 15 known (x, y) --> (u, v) mappings, we can now apply
        our regression model to the whole of the -5 to 5 grid we constructed above. By passing in
        the array `GridPoints` we are asking for our regression model to predict the x component of
        the velocity at each point in the grid.

        This is then replicated below for the y component of the velocity at each point in the grid.
    '''
    Ur, Ku = model_u.predict(GridPoints, full_cov=full)  # Kr = posterior covariance
    Vr, Kv = model_v.predict(GridPoints, full_cov=full)

    # Reshape the output velocity component matrices to be the same size and shape as
    # the inital matrices of x, y points
    ur = np.reshape(Ur, [y.size, x.size])
    vr = np.reshape(Vr, [y.size, x.size])

    ku = np.reshape(Ku, [y.size, x.size])
    kv = np.reshape(Kv, [y.size, x.size])

    return x, y, u, v, ur, vr, Xo, ku, kv, model_u, model_v