from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2019 Isuru Fernando"

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

from sumpy.tools import add_mi


class PDE(object):
    r"""
    Represents a iscalar PDEs of dimension `dim`. It is represented by a
    dictionary. A dictionary maps a a multi-index given as a tuple
    to the coefficient.
    """
    def __init__(self, dim, eq):
        """
        :arg dim: dimension of the PDE
        :arg eq: A dictionary mapping a multi-index to a value
        """
        self.dim = dim
        self.eq = eq

    def __mul__(self, param):
        res = PDE(self.dim, {})
        for k, v in self.eq.items():
            res.eq[k] = v * param
        return res

    __rmul__ = __mul__

    def __add__(self, other_pde):
        assert self.dim == other_pde.dim
        res = PDE(self.dim, self.eq.copy())
        for k, v in other_pde.eq.items():
            if k in res.eq:
                res.eq[k] += v
            else:
                res.eq[k] = v
        return res

    __radd__ = __add__

    def __sub__(self, other_pde):
        return self + (-1)*other_pde

    def __repr__(self):
        return repr(self.eq)


def laplacian(pde):
    dim = pde.dim
    res = PDE(dim, {})
    for j in range(dim):
        mi = [0]*dim
        mi[j] = 2
        res = res + diff(pde, tuple(mi))
    return res


def diff(pde, mi):
    res = PDE(pde.dim, {})
    for eq_mi, v in pde.eq.items():
        res.eq[add_mi(eq_mi, mi)] = v
    return res


def make_pde_sym(dim):
    """
    Returns a PDE u = 0
    """
    mi = tuple([0]*dim)
    return PDE(dim, {mi: 1})
