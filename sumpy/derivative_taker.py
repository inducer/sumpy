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
 .. autoclass:: ODEDerivativeTaker
 .. autoclass:: RadialDerivativeTaker
 .. autoclass:: DifferentiatedExprDerivativeTaker
"""

import logging
from dataclasses import dataclass, field
from math import factorial, prod
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from pytools import memoize_method

import sumpy.symbolic as sym
from sumpy.derivative_recurrence import derivative_recurrence
from sumpy.tools import add_mi, add_to_sac, mi_increment_axis, sub_mi


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import sympy as sp

    from sumpy.assignment_collection import SymbolicAssignmentCollection
    from sumpy.diff_op import MultiIndex


logger = logging.getLogger(__name__)


# {{{ ExprDerivativeTaker

@dataclass(frozen=True)
class ExprDerivativeTaker:
    r"""Facilitates the efficient computation of (potentially) high-order
    derivatives of a given :mod:`sympy` expression *expr* while attempting
    to maximize the number of common subexpressions generated.
    This class defines the interface and realizes a baseline implementation.
    More specialized implementations may offer better efficiency for special
    cases.

    A class to take scaled derivatives of the symbolic expression
    expr w.r.t. variables var_list and the scaling parameter rscale.

    Consider a Taylor multipole expansion:

    .. math::
        f (x - y) = \sum_{i = 0}^{\infty} (\partial_y^i f) (x - y) \big|_{y = c}
       \frac{(y - c)^i}{i!} .

    Now suppose we would like to use a scaled version :math:`g` of the
    kernel :math:`f`:

    .. math::

        \begin{aligned}
        f (x) & = g (x / \alpha),\\
        f^{(i)} (x) & = \frac{1}{\alpha^i} g^{(i)} (x / \alpha) .
        \end{aligned}

    where :math:`\alpha` is chosen to be on a length scale similar to
    :math:`x` (for example by choosing :math:`\alpha` proportional to the
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

        \begin{aligned}
        g (\xi) & = f (\alpha \xi),\\
        g^{(i)} (\xi) & = \alpha^i f^{(i)} (\alpha \xi) .
        \end{aligned}

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

    .. autoattribute:: orig_expr
    .. autoattribute:: var_list
    .. autoattribute:: rscale
    .. autoattribute:: sac

    .. automethod:: diff
    """
    orig_expr: sym.Expr
    var_list: Sequence[sym.Symbol]
    rscale: sym.Expr = field(default_factory=lambda: sym.sympify(1))
    sac: SymbolicAssignmentCollection | None = None

    cache_by_mi: dict[MultiIndex, sym.Expr] = field(init=False)

    def __post_init__(self):
        zero_mi = (0,) * self.dim
        object.__setattr__(self, "cache_by_mi", {zero_mi: self.orig_expr})

    @property
    def dim(self):
        return len(self.var_list)

    def diff(self, mi: MultiIndex) -> sym.Expr:
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

    def get_derivative_taking_sequence(self,
                start_mi: MultiIndex,
                end_mi: MultiIndex) -> Iterable[tuple[sym.Symbol, MultiIndex]]:
        current_mi = np.array(start_mi, dtype=int)
        for idx, (mi_i, vec_i) in enumerate(
                zip(sub_mi(end_mi, start_mi), self.var_list, strict=True)):
            for _ in range(1, 1 + mi_i):
                current_mi[idx] += 1
                yield vec_i, tuple(current_mi)

    def get_closest_cached_mi(self, mi: MultiIndex):
        return min((other_mi
                for other_mi in self.cache_by_mi
                if (np.array(mi) >= np.array(other_mi)).all()),
            key=lambda other_mi: sum(sub_mi(mi, other_mi)))


# }}}

# {{{ ODEDerivativeTaker

@dataclass(frozen=True)
class ODEDerivativeTaker(ExprDerivativeTaker):
    r"""
    Derivative taker for kernels :math:`G(z) = g(q)` with :math:`q = |z|^2`.

    The scalar function :math:`g` satisfies an ODE with polynomial coefficients,

    .. math::

        \sum_{i=0}^{n} a_i(q) g^{(i)}(q) = 0,
        \qquad g^{(i)}(q) := \left(\frac{d}{dq}\right)^i g(q),

    as returned by :func:`~sumpy.expansion.kernel_ode.kernel_ode`.

    For requested :math:`\partial_z^{\mathrm{mi}} G(z)`, the taker computes
    and caches any missing recurrence dependencies

    .. math::

        U[i, \mathrm{mi}] = \mathrm{rscale}^{|\mathrm{mi}|}
            \frac{\partial_z^{\mathrm{mi}} [g^{(i)}(|z|^2)]}{\mathrm{mi}!},
        \qquad 0 \le i < n,

    using :func:`~sumpy.derivative_recurrence.derivative_recurrence` along axis

    .. math::

        k = \begin{cases}
            0, & \mathrm{mi}_0 > 0, \\
            \operatorname{argmax}_j \mathrm{mi}_j, & \mathrm{mi}_0 = 0.
        \end{cases}

    :meth:`~ExprDerivativeTaker.diff` returns
    :math:`\mathrm{rscale}^{|\mathrm{mi}|} \partial_z^{\mathrm{mi}} G
    = \mathrm{mi}! U[0, \mathrm{mi}]`.
    """
    ode_coeffs: Sequence[sym.Expr] = ()
    init_conds: Sequence[sym.Expr] = ()

    cache_by_i_mi: dict[tuple[int, MultiIndex], sym.Expr] = field(init=False)

    @override
    def __post_init__(self):
        super().__post_init__()
        zero_mi = (0,) * self.dim
        states = {(0, zero_mi): self.orig_expr}
        for i, init_cond in enumerate(self.init_conds):
            if i > 0:
                states[i, zero_mi] = add_to_sac(self.sac, init_cond)
        object.__setattr__(self, "cache_by_i_mi", states)

    @memoize_method
    def _updates(self, axis: int):
        r"""
        Return the :math:`U[i, \mathrm{mi}]` update formula along *axis*.
        """
        return derivative_recurrence(self.ode_coeffs, self.dim, axis, self.rscale)

    def _state_mi(self, i: int, mi: MultiIndex) -> sym.Expr:
        r"""
        Return :math:`U[i, \mathrm{mi}]
        = [h^{\mathrm{mi}}] g^{(i)}(|z + \mathrm{rscale} \cdot h|^2)`.
        """
        try:
            return self.cache_by_i_mi[i, mi]
        except KeyError:
            pass

        # For {mi: mi_0 > 0}:
        # U[i, mi] -> (U[i+1, mi-e_0], U[i+1, mi-2e_0])
        #          -> (U[i+2, mi-2e_0], U[i+2, mi-3e_0], U[i+2, mi-4e_0])
        #
        # For {mi: mi_0 = 0}, update along the axis = argmax_k mi_k

        axis = 0 if mi[0] else max(range(self.dim), key=lambda k: mi[k])

        subst = {
            **{sym.Symbol(f"z{k}"): v for k, v in enumerate(self.var_list)},
            **{sym.Symbol(f"mi{k}"): index for k, index in enumerate(mi)}}
        expr = 0
        for (prev_state, shift), weight in self._updates(axis)[i].items():
            prev_mi = add_mi(mi, shift)
            if min(prev_mi) < 0:
                continue
            expr += weight.xreplace(subst)*self._state_mi(prev_state, prev_mi)
        expr = add_to_sac(self.sac, expr)
        self.cache_by_i_mi[i, mi] = expr
        return expr

    @override
    def diff(self, mi: MultiIndex) -> sym.Expr:
        return prod(map(factorial, mi))*self._state_mi(0, mi)


# }}}

# {{{ RadialDerivativeTaker

@dataclass(frozen=True)
class RadialDerivativeTaker(ExprDerivativeTaker):
    r"""Derivative taker for radial kernel :math:`G(z) = g(q)`,
    :math:`q = |z|^2`, using Tausch's recurrence [Tausch2003]_.

    Define :math:`g^{(i)} := d^i g/dq^i` and :math:`g_i := 2^i g^{(i)}(q)`,
    so that :math:`\partial_z^{e_k} g_i = z_k g_{i+1}`.
    For any axis :math:`k` with :math:`\mathrm{mi}_k \ge 1`,

    .. math::

        \partial_z^{\mathrm{mi}} g_i
            = \partial_z^{\mathrm{mi} - e_k} (z_k g_{i+1})
            = z_k \partial_z^{\mathrm{mi} - e_k} g_{i+1}
            + (\mathrm{mi}_k - 1) \partial_z^{\mathrm{mi} - 2 e_k} g_{i+1}.

    The scaled derivatives :math:`T_{i,\mathrm{mi}} =
    \mathrm{rscale}^{|\mathrm{mi}|} \partial_z^{\mathrm{mi}} g_i` satisfy the
    recurrence relation

    .. math::

        T_{i,\mathrm{mi}}
            = \mathrm{rscale} \cdot z_k T_{i+1,\mathrm{mi} - e_k}
            + (\mathrm{mi}_k - 1) \mathrm{rscale}^2 T_{i+1,\mathrm{mi} - 2 e_k},
        \qquad T_{i,0} = g_i.

    :meth:`~ExprDerivativeTaker.diff` returns :math:`T_{0,\mathrm{mi}}`. Over all
    derivatives up to total order :math:`p`, the amortized cost is :math:`O(p)`
    per derivative.

    .. [Tausch2003] Tausch, J., 2003. The fast multipole method for
       arbitrary Green's functions. Contemporary Mathematics, 329.
    """

    cache_by_i_mi: dict[tuple[int, MultiIndex], sym.Expr] = field(init=False)

    @override
    def __post_init__(self):
        super().__post_init__()
        zero_mi = (0,) * len(self.var_list)
        object.__setattr__(self, "cache_by_i_mi", {(0, zero_mi): self.orig_expr})

    @property
    @memoize_method
    def q(self):
        return sum(v**2 for v in self.var_list)

    @property
    @memoize_method
    def radial_expr(self):
        argument = [sym.sqrt(sym.Symbol("q")), *([0]*(self.dim - 1))]
        return self.orig_expr.subs(dict(zip(self.var_list, argument, strict=True)))

    @property
    @memoize_method
    def is_radial(self):
        radial = self.radial_expr.xreplace({sym.Symbol("q"): self.q})
        return (self.orig_expr - radial).expand() == 0

    @property
    @memoize_method
    def var_list_multiplied(self):
        return [add_to_sac(self.sac, v * self.rscale) for v in self.var_list]

    def diff(self, mi: MultiIndex, i: int = 0) -> sym.Expr:
        r"""
        Return :math:`T_{i,\mathrm{mi}}` by the recurrence of [Tausch2003]_. The
        default :math:`i = 0` gives the derivatives of :math:`G`. An
        expression that is not radial is differentiated directly.
        """
        if not self.is_radial:
            assert i == 0
            return ExprDerivativeTaker.diff(self, mi)

        try:
            return self.cache_by_i_mi[i, mi]
        except KeyError:
            pass

        for axis in range(self.dim):
            if mi[axis] == 1:
                mi_minus_one = mi_increment_axis(mi, axis, -1)
                # T_{i,mi} = rscale*z_k*T_{i+1,mi-e_k}, the mi_k = 1 case.
                expr = self.var_list_multiplied[axis] * self.diff(mi_minus_one, i+1)
                self.cache_by_i_mi[i, mi] = expr
                return expr

        for axis in range(self.dim):
            if mi[axis] >= 2:
                mi_minus_one = mi_increment_axis(mi, axis, -1)
                mi_minus_two = mi_increment_axis(mi, axis, -2)
                # T_{i,mi} = rscale*z_k*T_{i+1,mi-e_k}
                #            + (mi_k - 1)*rscale^2*T_{i+1,mi-2e_k}
                expr = self.var_list_multiplied[axis] * self.diff(mi_minus_one, i+1)
                expr += (mi[axis]-1)*self.diff(mi_minus_two, i+1) * self.rscale ** 2
                expr = add_to_sac(self.sac, expr)
                self.cache_by_i_mi[i, mi] = expr
                return expr

        zero_mi = (0,) * self.dim
        assert mi == zero_mi
        assert i > 0

        # Step 1 of [Tausch2003]:
        # g_i = 2^i g^(i)(q), with q = |var_list|^2.
        q = sym.Symbol("q")
        expr = 2**i * self.radial_expr.diff(q, i).xreplace({q: self.q})
        self.cache_by_i_mi[i, zero_mi] = expr
        return expr


# }}}

# {{{ DifferentiatedExprDerivativeTaker

DerivativeCoeffDict = dict[tuple[int, ...], int | float | complex | sym.Expr]


@dataclass(frozen=True)
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

    def diff(self, mi: MultiIndex, save_intermediate=lambda x: x) -> sym.Expr:
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

def diff_derivative_coeff_dict(
        derivative_coeff_dict: DerivativeCoeffDict,
        variable_idx: int,
        variables: sp.Matrix) -> DerivativeCoeffDict:
    """Differentiate a derivative transformation dictionary given by
    *derivative_coeff_dict* using the variable given by **variable_idx**
    and return a new derivative transformation dictionary.
    """
    from collections import defaultdict
    new_derivative_coeff_dict: DerivativeCoeffDict = defaultdict(lambda: 0)

    for mi, coeff in derivative_coeff_dict.items():
        # In the case where we have x * u.diff(x), the result should be
        #       x.diff(x) + x * u.diff(x, x)
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
