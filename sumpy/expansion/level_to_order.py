__copyright__ = "Copyright (C) 2016 Matt Wala"

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

__doc__ = """
.. autoclass:: FMMLibExpansionOrderFinder
.. autoclass:: SimpleExpansionOrderFinder
"""

import math

import numpy as np


class FMMLibExpansionOrderFinder:
    r"""Return expansion orders that meet the tolerance for a given level
    using routines wrapped from ``pyfmmlib``.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, tol, extra_order=0):
        """
        :arg tol: error tolerance
        :arg extra_order: order increase to accommodate, say, the taking of
            derivatives of the FMM expansions.
        """
        self.tol = tol
        self.extra_order = extra_order

    def __call__(self, kernel, kernel_args, tree, level):
        from pyfmmlib import (          # pylint: disable=no-name-in-module
            l2dterms, l3dterms, h2dterms, h3dterms)
        from sumpy.kernel import LaplaceKernel, HelmholtzKernel

        if isinstance(kernel, LaplaceKernel):
            if tree.dimensions == 2:
                nterms, ier = l2dterms(self.tol)
                if ier:
                    raise RuntimeError(f"l2dterms returned error code '{ier}'")

            elif tree.dimensions == 3:
                nterms, ier = l3dterms(self.tol)
                if ier:
                    raise RuntimeError(f"l3dterms returned error code '{ier}'")

            else:
                raise ValueError(f"unsupported dimension: {tree.dimensions}")

        elif isinstance(kernel, HelmholtzKernel):
            helmholtz_k = dict(kernel_args)[kernel.helmholtz_k_name]
            size = tree.root_extent / 2 ** level

            if tree.dimensions == 2:
                nterms, ier = h2dterms(size, helmholtz_k, self.tol)
                if ier:
                    raise RuntimeError(f"h2dterms returned error code '{ier}'")

            elif tree.dimensions == 3:
                nterms, ier = h3dterms(size, helmholtz_k, self.tol)
                if ier:
                    raise RuntimeError(f"h3dterms returned error code '{ier}'")

            else:
                raise ValueError(f"unsupported dimension: {tree.dimensions}")

        else:
            raise TypeError(f"unsupported kernel: '{type(kernel).__name__}'")

        return nterms + self.extra_order


class SimpleExpansionOrderFinder:
    r"""
    This models the Laplace truncation error as:

    .. math::

        C_{\text{lap}} \left(\frac{\sqrt{d}}{3}\right)^{p+1}.

    For the Helmholtz kernel, an additional term is added:

    .. math::

        C_{\text{helm}} \frac 1{p!}
        \left(C_{\text{helmscale}} \cdot \frac{hk}{2\pi}\right)^{p+1},

    where :math:`d` is the number of dimensions, :math:`p` is the expansion order,
    :math:`h` is the box size, and :math:`k` is the wave number.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, tol, err_const_laplace=0.01, err_const_helmholtz=100,
            scaling_const_helmholtz=4,
            extra_order=1):
        """
        :arg extra_order: order increase to accommodate, say, the taking of
            derivatives of the FMM expansions.
        """
        self.tol = tol

        self.err_const_laplace = err_const_laplace
        self.err_const_helmholtz = err_const_helmholtz
        self.scaling_const_helmholtz = scaling_const_helmholtz

        self.extra_order = extra_order

    def __call__(self, kernel, kernel_args, tree, level):
        from sumpy.kernel import LaplaceKernel, HelmholtzKernel

        assert isinstance(kernel, (LaplaceKernel, HelmholtzKernel))

        laplace_order = int(np.ceil(
                (np.log(self.tol) - np.log(self.err_const_laplace))
                /
                np.log(
                    np.sqrt(tree.dimensions)/3
                    ) - 1))

        if isinstance(kernel, HelmholtzKernel):
            helmholtz_k = dict(kernel_args)[kernel.helmholtz_k_name]

            box_lengthscale = (
                tree.stick_out_factor
                * tree.root_extent / (1 << level))

            factor = (
                    self.scaling_const_helmholtz
                    * box_lengthscale
                    * helmholtz_k
                    / (2*float(np.pi)))

            helm_order = 1
            helm_rec_error = self.err_const_helmholtz * factor
            while True:
                helm_rec_error = helm_rec_error * factor / (helm_order+1)

                if helm_order < 4:
                    # this may overflow for large orders
                    helm_error_direct = (
                            1/math.factorial(helm_order+1)
                            * self.err_const_helmholtz
                            * factor**(helm_order+1))
                    assert (abs(helm_rec_error - helm_error_direct)
                            < 1e-13 * abs(helm_error_direct))

                if helm_rec_error * helm_order**(tree.dimensions-1) < self.tol:
                    break

                helm_order += 1

                if helm_order > 10000:
                    raise ValueError("unable to find suitable order estimate "
                            "for Helmholtz expansion")
        else:
            helm_order = 0

        return max(laplace_order, helm_order) + self.extra_order


# vim: fdm=marker
