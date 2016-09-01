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


# vim: fdm=marker
