from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2017 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
import numpy.linalg as la
from pytools import memoize_method

__doc__ = """
.. autoclass:: CalculusPatch

.. autofunction:: frequency_domain_maxwell
"""


class CalculusPatch(object):
    """Sets up a grid of points on which derivatives can be calculated. Useful
    to verify that an evaluated potential actually solves a PDE.

    .. attribute: dim

    .. attribute: points

        shape: ``(dim, npoints_total)``

    .. automethod:: weights
    .. automethod:: basis
    .. automethod:: diff
    .. automethod:: dx
    .. automethod:: dy
    .. automethod:: dy
    .. automethod:: laplace
    .. automethod:: div
    .. automethod:: curl
    .. automethod:: eval_at_center
    .. autoattribute:: x
    .. autoattribute:: y
    .. autoattribute:: z
    .. automethod:: norm
    .. automethod:: plot_nodes
    .. automethod:: plot
    """
    def __init__(self, center, h=1e-1, order=4, nodes="chebyshev"):
        self.center = center

        npoints = order + 1
        if nodes == "equispaced":
            points_1d = np.linspace(-h/2, h/2, npoints)
            weights_1d = None

        elif nodes == "chebyshev":
            a = np.arange(npoints, dtype=np.float64)
            points_1d = (h/2)*np.cos((2*(a+1)-1)/(2*npoints)*np.pi)
            weights_1d = None

        elif nodes == "legendre":
            from scipy.special import legendre
            points_1d, weights_1d, _ = legendre(npoints).weights.T
            points_1d = points_1d * (h/2)
            weights_1d = weights_1d * (h/2)

        else:
            raise ValueError("invalid node set: %s" % nodes)

        self.h = h
        self.npoints = npoints
        self._points_1d = points_1d
        self._weights_1d = weights_1d

        self.dim = dim = len(self.center)
        self.center = center

        points_shaped = np.array(np.meshgrid(
                *[center[i] + points_1d for i in range(dim)],
                indexing="ij"))

        self._points_shaped = points_shaped
        self.points = points_shaped.reshape(dim, -1)

        self._pshape = points_shaped.shape[1:]

    @memoize_method
    def _vandermonde_1d(self):
        points_1d = self._points_1d

        npoints = len(self._points_1d)
        vandermonde = np.zeros((npoints, npoints))
        for i in range(npoints):
            vandermonde[:, i] = points_1d**i

        return vandermonde

    @memoize_method
    def _zero_eval_vec_1d(self):
        # The zeroth coefficient--all others involve x=0.
        return self._vandermonde_1d()[0]

    def basis(self):
        """"
        :returns: a :class:`list` containing functions that realize
            a high-order interpolation basis on the :attr:`points`.
        """

        from pytools import indices_in_shape
        from scipy.special import eval_chebyt

        def eval_basis(ind, x):
            result = 1
            for i in range(self.dim):
                coord = (x[i] - self.center[i])/(self.h/2)
                result *= eval_chebyt(ind[i], coord)
            return result

        from functools import partial
        return [
                partial(eval_basis, ind)
                for ind in indices_in_shape((self.npoints,)*self.dim)]

    @memoize_method
    def weights(self):
        """"
        :returns: a vector of high-order quadrature weights on the :attr:`points`
        """

        if self._weights_1d is None:
            raise NotImplementedError("weights not available for these nodes")

        result = np.ones_like(self._points_shaped[0])
        for i in range(self.dim):
            result = result * self._weights_1d[
                    (slice(None),)
                    + i*(np.newaxis,)]

        return result.reshape(-1)

    @memoize_method
    def _diff_mat_1d(self, nderivs):
        npoints = len(self._points_1d)

        vandermonde = self._vandermonde_1d()
        coeff_diff_mat = np.diag(np.arange(1, npoints), 1)

        n_diff_mat = np.eye(npoints)
        for i in range(nderivs):
            n_diff_mat = n_diff_mat.dot(coeff_diff_mat)

        deriv_coeffs_mat = la.solve(vandermonde.T, n_diff_mat.T).T
        return vandermonde.dot(deriv_coeffs_mat)

    def diff(self, axis, f_values, nderivs=1):
        """Return the derivative along *axis* of *f_values*.

        :arg f_values: an array of shape ``(npoints_total,)``
        :returns: an array of shape ``(npoints_total,)``
        """

        from numbers import Number
        if isinstance(f_values, (np.number, Number)):
            # constants differentiate to 0
            return 0

        dim = len(self.center)

        assert axis < dim

        axes = "klmno"
        src_axes = (axes[:axis] + "j" + axes[axis:])[:dim]
        tgt_axes = (axes[:axis] + "i" + axes[axis:])[:dim]

        return np.einsum(
                "ij,%s->%s" % (src_axes, tgt_axes),
                self._diff_mat_1d(nderivs),
                f_values.reshape(*self._pshape)).reshape(-1)

    def dx(self, f_values):
        return self.diff(0, f_values)

    def dy(self, f_values):
        return self.diff(1, f_values)

    def dz(self, f_values):
        return self.diff(2, f_values)

    def laplace(self, f_values):
        """Return the Laplacian of *f_values*.

        :arg f_values: an array of shape ``(npoints_total,)``
        :returns: an array of shape ``(npoints_total,)``
        """

        return sum(self.diff(iaxis, f_values, 2) for iaxis in range(self.dim))

    def div(self, arg):
        """
        :arg arg: an object array containing
            :class:`numpy.ndarrays` with shape ``(npoints_total,)``.
        """
        result = 0
        for i, arg_i in enumerate(arg):
            result = result + self.diff(i, arg_i)

        return result

    def curl(self, arg):
        """Take the curl of the vector quantity *arg*.

        :arg arg: an object array of shape ``(3,)`` containing
            :class:`numpy.ndarrays` with shape ``(npoints_total,)``.
        """
        from pytools import levi_civita
        from pytools.obj_array import make_obj_array
        return make_obj_array([
            sum(
                levi_civita((k, m, n)) * self.diff(m, arg[n])
                for m in range(3) for n in range(3))
            for k in range(3)])

    def eval_at_center(self, f_values):
        """Interpolate *f_values* to the center point.

        :arg f_values: an array of shape ``(npoints_total,)``
        :returns: a scalar.
        """
        f_values = f_values.reshape(*self._pshape)
        for i in range(self.dim):
            f_values = self._zero_eval_vec_1d.dot(f_values)

        return f_values

    @property
    def x(self):
        return self.points[0]

    @property
    def y(self):
        return self.points[1]

    @property
    def z(self):
        return self.points[2]

    def norm(self, arg, p):
        if p == np.inf:
            if arg.dtype == np.object:
                return max(
                        la.norm(x_i, p)
                        for x_i in arg)
            else:
                return la.norm(arg, p)
        else:
            raise ValueError("unsupported norm")

    def plot_nodes(self):
        import matplotlib.pyplot as plt
        plt.gca().set_aspect("equal")
        plt.plot(
            self._points_shaped[0].reshape(-1),
            self._points_shaped[1].reshape(-1),
            "o")

    def plot(self, f):
        f = f.reshape(*self._pshape)

        import matplotlib.pyplot as plt
        plt.gca().set_aspect("equal")
        plt.contourf(self._points_1d, self._points_1d, f)


def frequency_domain_maxwell(cpatch, e, h, k):
    mu = 1
    epsilon = 1
    c = 1/np.sqrt(mu*epsilon)
    omega = k*c

    b = mu*h
    d = epsilon*e

    # https://en.wikipedia.org/w/index.php?title=Maxwell%27s_equations&oldid=798940325#Macroscopic_formulation
    # assumed time dependence exp(-1j*omega*t)
    # Agrees with Jackson, Third Ed., (8.16)
    resid_faraday = cpatch.curl(e) - 1j * omega * b
    resid_ampere = cpatch.curl(h) + 1j * omega * d

    resid_div_e = cpatch.div(e)
    resid_div_h = cpatch.div(h)

    return (
            resid_faraday,
            resid_ampere,
            resid_div_e,
            resid_div_h)
