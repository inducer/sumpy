from __future__ import division

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
.. autofunction:: h2d_level_to_order_lookup
.. autofunction:: l2d_level_to_order_lookup

.. autoclass:: SimpleExpansionOrderFinder
"""

import numpy as np


def h2d_level_to_order_lookup(tree, helmholtz_k, epsilon):
    """
    Compute a mapping from level number to expansion order,
    Helmholtz 2D case.

    This wraps the function *h2dterms* from :mod:`pyfmmlib`.

    :arg tree: An instance of :class:`boxtree.Tree`.
    :arg helmholtz_k: Helmholtz parameter
    :arg epsilon: Precision

    :return: A :class:`numpy.array` of length `tree.nlevels`
    """

    if tree.dimensions != 2:
        raise ValueError("tree must be 2d")

    orders = np.empty(tree.nlevels, dtype=int)
    bbox_area = np.max(
        tree.bounding_box[1] - tree.bounding_box[0]) ** 2

    from pyfmmlib import h2dterms
    for level in range(tree.nlevels):
        nterms, ier = h2dterms(bbox_area / 2 ** level, helmholtz_k, epsilon)
        if ier != 0:
            raise RuntimeError(
                "h2dterms returned error code {ier}".format(ier=ier))
        orders[level] = nterms

    return orders


def l2d_level_to_order_lookup(tree, epsilon):
    """
    Compute a mapping from level number to expansion order,
    Laplace 2D case.

    This wraps the function *l2dterms* from :mod:`pyfmmlib`.

    :arg tree: An instance of :class:`boxtree.Tree`.
    :arg epsilon: Precision

    :return: A :class:`numpy.array` of length `tree.nlevels`
    """

    if tree.dimensions != 2:
        raise ValueError("tree must be 2d")

    from pyfmmlib import l2dterms
    nterms, ier = l2dterms(epsilon)
    if ier != 0:
        raise RuntimeError(
            "l2dterms returned error code {ier}".format(ier=ier))

    orders = np.empty(tree.nlevels, dtype=int)
    orders.fill(nterms)

    return orders


class SimpleExpansionOrderFinder(object):
    r"""
    This models the Laplace truncation error as:

        C_{\text{lap}} \left(\frac{\sqrt{d}}{3}\right)^{p+1}.

    For the Helmholtz kernel, an additional term is added:

    .. math::

        C_{\text{helm}} \frac 1{p!}
        \left(C_{\text{helm\_scale}} \cdot \frac{hk}{2\pi}\right)^{p+1},

    where :math:`d` is the number of dimensions, :math:`p` is the expansion order,
    :math:`h` is the box size, and :math:`k` is the wave number.
    """

    def __init__(self, tol, err_const_laplace=0.01, err_const_helmholtz=100,
            scaling_const_helmholtz=4,
            extra_order=1):
        """
        :arg extra_order: order increase to accommodate, say, the taking of
            oderivatives f the FMM expansions.
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

            from math import factorial
            helm_order = 1
            helm_rec_error = self.err_const_helmholtz * factor
            while True:
                helm_rec_error = helm_rec_error * factor / (helm_order+1)

                if helm_order < 4:
                    # this may overflow for large orders
                    helm_error_direct = (
                            1/factorial(helm_order+1)
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
