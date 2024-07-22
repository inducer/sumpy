from __future__ import annotations


__copyright__ = """
Copyright (C) 2012 Andreas Kloeckner
Copyright (C) 2020 Isuru Fernando
"""

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

 Derivative Taker
 ================

 .. autoclass:: ExprDerivativeTaker
 .. autoclass:: LaplaceDerivativeTaker
 .. autoclass:: RadialDerivativeTaker
 .. autoclass:: HelmholtzDerivativeTaker
 .. autoclass:: DifferentiatedExprDerivativeTaker
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from pytools.tag import tag_dataclass

import sumpy.symbolic as sym
from sumpy.tools import add_mi, add_to_sac


logger = logging.getLogger(__name__)


# {{{ ExprDerivativeTaker

class ExprDerivativeTaker:
    """Facilitates the efficient computation of (potentially) high-order
    derivatives of a given :mod:`sympy` expression *expr* while attempting
    to maximize the number of common subexpressions generated.
    This class defines the interface and realizes a baseline implementation.
    More specialized implementations may offer better efficiency for special
    cases.
    .. automethod:: diff
    """

    def __init__(self, expr, var_list, rscale=1, sac=None):
        r"""
        A class to take scaled derivatives of the symbolic expression
        expr w.r.t. variables var_list and the scaling parameter rscale.
        Consider a Taylor multipole expansion:
        .. math::
            f (x - y) = \sum_{i = 0}^{\infty} (\partial_y^i f) (x - y) \big|_{y = c}
           \frac{(y - c)^i}{i!} .
        Now suppose we would like to use a scaled version :math:`g` of the
        kernel :math:`f`:
        .. math::
            \begin{eqnarray*}
              f (x) & = & g (x / \alpha),\\
              f^{(i)} (x) & = & \frac{1}{\alpha^i} g^{(i)} (x / \alpha) .
            \end{eqnarray*}
        where :math:`\alpha` is chosen to be on a length scale similar to
        :math:`x` (for example by choosing :math:`\alpha` proporitional to the
        size of the box for which the expansion is intended) so that :math:`x /
        \alpha` is roughly of unit magnitude, to avoid arithmetic issues with
        small arguments. This yields
        .. math::
            f (x - y) = \sum_{i = 0}^{\infty} (\partial_y^i g)
            \left( \frac{x - y}{\alpha} \right) \Bigg|_{y = c}
            \cdot
            \frac{(y - c)^i}{\alpha^i \cdot i!}.
        Observe that the :math:`(y - c)` term is now scaled to unit magnitude,
        as is the argument of :math:`g`.
        With :math:`\xi = x / \alpha`, we find
        .. math::
            \begin{eqnarray*}
              g (\xi) & = & f (\alpha \xi),\\
              g^{(i)} (\xi) & = & \alpha^i f^{(i)} (\alpha \xi) .
            \end{eqnarray*}
        Generically for all kernels, :math:`f^{(i)} (\alpha \xi)` is computable
        by taking a sufficient number of symbolic derivatives of :math:`f` and
        providing :math:`\alpha \xi = x` as the argument.
        Now, for some kernels, like :math:`f (x) = C \log x`, the powers of
        :math:`\alpha^i` from the chain rule cancel with the ones from the
        argument substituted into the kernel derivatives:
        .. math::
            g^{(i)} (\xi) = \alpha^i f^{(i)} (\alpha \xi) = C' \cdot \alpha^i \cdot
            \frac{1}{(\alpha x)^i} \quad (i > 0),
        making them what you might call *scale-invariant*.
        This derivative taker returns :math:`g^{(i)}(\xi) = \alpha^i f^{(i)}`
        given :math:`f^{(0)}` as *expr* and :math:`\alpha` as :attr:`rscale`.
        """

        assert isinstance(expr, sym.Basic)
        self.var_list = var_list
        zero_mi = (0,) * len(var_list)
        self.cache_by_mi = {zero_mi: expr}
        self.rscale = rscale
        self.sac = sac
        self.dim = len(self.var_list)
        self.orig_expr = expr

    def mi_dist(self, a, b):
        return np.array(a, dtype=int) - np.array(b, dtype=int)

    def diff(self, mi):
        """Take the derivative of the expression represented by
        :class:`ExprDerivativeTaker`.
        :param mi: multi-index representing the derivative
        """
        try:
            return self.cache_by_mi[mi]
        except KeyError:
            pass

        current_mi = self.get_closest_cached_mi(mi)
        expr = self.cache_by_mi[current_mi]

        for next_deriv, next_mi in self.get_derivative_taking_sequence(
                current_mi, mi):
            expr = expr.diff(next_deriv) * self.rscale
            self.cache_by_mi[next_mi] = expr

        return expr

    def get_derivative_taking_sequence(self, start_mi, end_mi):
        current_mi = np.array(start_mi, dtype=int)
        for idx, (mi_i, vec_i) in enumerate(
                zip(self.mi_dist(end_mi, start_mi), self.var_list)):
            for _ in range(1, 1 + mi_i):
                current_mi[idx] += 1
                yield vec_i, tuple(current_mi)

    def get_closest_cached_mi(self, mi):
        return min((other_mi
                for other_mi in self.cache_by_mi
                if (np.array(mi) >= np.array(other_mi)).all()),
            key=lambda other_mi: sum(self.mi_dist(mi, other_mi)))


# }}}

# {{{ LaplaceDerivativeTaker

class LaplaceDerivativeTaker(ExprDerivativeTaker):
    """Specialized derivative taker for Laplace potential.
    """

    def __init__(self, expr, var_list, rscale=1, sac=None):
        super().__init__(expr, var_list, rscale, sac)
        self.scaled_var_list = [add_to_sac(self.sac, v/rscale) for v in var_list]
        self.scaled_r = add_to_sac(self.sac,
                sym.sqrt(sum(v**2 for v in self.scaled_var_list)))

    def diff(self, mi):
        """
        Implements the algorithm described in [Fernando2021] to take cartesian
        derivatives of Laplace potential using recurrences. Cost of each derivative
        is amortized constant.
        .. [Fernando2021]: Fernando, I., Klöckner, A., 2021. Automatic Synthesis of
                           Low Complexity Translation Operators for the Fast
                           Multipole Method. In preparation.
        """
        # Return zero for negative values. Makes the algorithm readable.
        if min(mi) < 0:
            return 0
        try:
            return self.cache_by_mi[mi]
        except KeyError:
            pass

        dim = self.dim
        if max(mi) == 1:
            return ExprDerivativeTaker.diff(self, mi)
        d = -1
        for i in range(dim):
            if mi[i] >= 2:
                d = i
                break
        assert d >= 0
        expr = 0
        for i in range(dim):
            mi_minus_one = list(mi)
            mi_minus_one[i] -= 1
            mi_minus_one = tuple(mi_minus_one)
            mi_minus_two = list(mi)
            mi_minus_two[i] -= 2
            mi_minus_two = tuple(mi_minus_two)
            x = self.scaled_var_list[i]
            n = mi[i]
            if i == d:
                if dim == 3:
                    expr -= (2*n - 1) * x * self.diff(mi_minus_one)
                    expr -= (n - 1)**2 * self.diff(mi_minus_two)
                else:
                    expr -= 2 * x * (n - 1) * self.diff(mi_minus_one)
                    expr -= (n - 1) * (n - 2) * self.diff(mi_minus_two)
                    if n == 2 and sum(mi) == 2:
                        expr += 1
            else:
                expr -= 2 * n * x * self.diff(mi_minus_one)
                expr -= n * (n - 1) * self.diff(mi_minus_two)
        expr /= self.scaled_r**2
        expr = add_to_sac(self.sac, expr)
        self.cache_by_mi[mi] = expr
        return expr


# }}}

# {{{ RadialDerivativeTaker

class RadialDerivativeTaker(ExprDerivativeTaker):
    """Specialized derivative taker for radial expressions.
    """

    def __init__(self, expr, var_list, rscale=1, sac=None):
        """
        Takes the derivatives of a radial function.
        """
        import sumpy.symbolic as sym
        super().__init__(expr, var_list, rscale, sac)
        empty_mi = (0,) * len(var_list)
        self.cache_by_mi_q = {(empty_mi, 0): expr}
        self.r = sym.sqrt(sum(v**2 for v in var_list))
        rsym = sym.Symbol("_r")
        r_expr = expr.xreplace({self.r**2: rsym**2})
        self.is_radial = not any(r_expr.has(v) for v in var_list)
        self.var_list_multiplied = [add_to_sac(sac, v * rscale) for v in var_list]

    def diff(self, mi, q=0):
        """
        Implements the algorithm described in [Tausch2003] to take cartesian
        derivatives of radial functions using recurrences. Cost of each derivative
        is amortized linear in the degree.
        .. [Tausch2003]: Tausch, J., 2003. The fast multipole method for arbitrary
                         Green's functions.
                         Contemporary Mathematics, 329, pp.307-314.
        """
        if not self.is_radial:
            assert q == 0
            return ExprDerivativeTaker.diff(self, mi)

        try:
            return self.cache_by_mi_q[(mi, q)]
        except KeyError:
            pass

        for i in range(self.dim):
            if mi[i] == 1:
                mi_minus_one = list(mi)
                mi_minus_one[i] = 0
                mi_minus_one = tuple(mi_minus_one)
                expr = self.var_list_multiplied[i] * self.diff(mi_minus_one, q=q+1)
                self.cache_by_mi_q[(mi, q)] = expr
                return expr

        for i in range(self.dim):
            if mi[i] >= 2:
                mi_minus_one = list(mi)
                mi_minus_one[i] -= 1
                mi_minus_one = tuple(mi_minus_one)
                mi_minus_two = list(mi)
                mi_minus_two[i] -= 2
                mi_minus_two = tuple(mi_minus_two)
                expr = (mi[i]-1)*self.diff(mi_minus_two, q=q+1) * self.rscale ** 2
                expr += self.var_list_multiplied[i] * self.diff(mi_minus_one, q=q+1)
                expr = add_to_sac(self.sac, expr)
                self.cache_by_mi_q[(mi, q)] = expr
                return expr

        assert mi == (0,)*self.dim
        assert q > 0

        prev_expr = self.diff(mi, q=q-1)
        # Need to get expr.diff(r)/r, but we can only do expr.diff(x)
        # Use expr.diff(x) = expr.diff(r) * x / r
        expr = prev_expr.diff(self.var_list[0])/self.var_list[0]
        # We need to distribute the division above
        expr = expr.expand(deep=False)
        self.cache_by_mi_q[(mi, q)] = expr
        return expr


# }}}

# {{{ HelmholtzDerivativeTaker

class HelmholtzDerivativeTaker(RadialDerivativeTaker):
    """Specialized derivative taker for Helmholtz potential.
    """

    def diff(self, mi, q=0):
        import sumpy.symbolic as sym
        if q < 2 or mi != (0,)*self.dim:
            return RadialDerivativeTaker.diff(self, mi, q)

        try:
            return self.cache_by_mi_q[(mi, q)]
        except KeyError:
            pass

        if self.dim == 2:
            # See https://dlmf.nist.gov/10.6.E6
            # and https://dlmf.nist.gov/10.6#E1
            k = self.orig_expr.args[1] / self.r
            expr = (-2*(q - 1) * self.diff(mi, q - 1)
                    - k**2 * self.diff(mi, q - 2)) / self.r**2
        else:
            # See reference [Tausch2003] in RadialDerivativeTaker.diff
            # Note that there is a typo in the paper where
            # -k**2/r is given instead of -k**2/r**2.
            k = (self.orig_expr * self.r).args[-1] / sym.I / self.r
            expr = (-(2*q - 1) * self.diff(mi, q - 1)
                    - k**2 * self.diff(mi, q - 2)) / self.r**2
        self.cache_by_mi_q[(mi, q)] = expr
        return expr


# }}}

# {{{ DifferentiatedExprDerivativeTaker

DerivativeCoeffDict = Dict[Tuple[int, ...], Any]


@tag_dataclass
class DifferentiatedExprDerivativeTaker:
    """Implements the :class:`ExprDerivativeTaker` interface
    for an expression that is itself a linear combination of
    derivatives of a base expression. To take the actual derivatives,
    it makes use of an underlying derivative taker *taker*.

    .. attribute:: taker
        A :class:`ExprDerivativeTaker` for the base expression.

    .. attribute:: derivative_coeff_dict
        A dictionary mapping a derivative multi-index to a coefficient.
        The expression represented by this derivative taker is the linear
        combination of the derivatives of the expression for the
        base expression.
    """
    taker: ExprDerivativeTaker
    derivative_coeff_dict: DerivativeCoeffDict

    def diff(self, mi, save_intermediate=lambda x: x):
        # By passing `rscale` to the derivative taker we are taking a scaled
        # version of the derivative which is `expr.diff(mi)*rscale**sum(mi)`
        # which might be implemented efficiently for kernels like Laplace.
        # One caveat is that we are taking more derivatives because of
        # :attr:`derivative_coeff_dict` which would multiply the
        # expression by more `rscale`s than necessary. This is corrected by
        # dividing by `rscale`.
        max_order = max(
                sum(extra_mi) for extra_mi in self.derivative_coeff_dict
                )

        result = sum(
            coeff * self.taker.diff(add_mi(mi, extra_mi))
            / self.taker.rscale ** (sum(extra_mi) - max_order)
            for extra_mi, coeff in self.derivative_coeff_dict.items())

        return result * save_intermediate(1 / self.taker.rscale ** max_order)


# }}}

# {{{ Helper functions

def diff_derivative_coeff_dict(derivative_coeff_dict: DerivativeCoeffDict,
        variable_idx, variables):
    """Differentiate a derivative transformation dictionary given by
    *derivative_coeff_dict* using the variable given by **variable_idx**
    and return a new derivative transformation dictionary.
    """
    from collections import defaultdict
    new_derivative_coeff_dict: DerivativeCoeffDict = defaultdict(lambda: 0)

    for mi, coeff in derivative_coeff_dict.items():
        # In the case where we have x * u.diff(x), the result should
        # be x.diff(x) + x * u.diff(x, x)
        # Calculate the first term by differentiating the coefficients
        new_coeff = sym.sympify(coeff).diff(variables[variable_idx])
        new_derivative_coeff_dict[mi] += new_coeff
        # Next calculate the second term by differentiating the derivatives
        new_mi = list(mi)
        new_mi[variable_idx] += 1
        new_derivative_coeff_dict[tuple(new_mi)] += coeff
    return {derivative: coeff for derivative, coeff in
            new_derivative_coeff_dict.items() if coeff != 0}

# }}}

# vim: fdm=marker
