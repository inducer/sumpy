from __future__ import annotations


__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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

import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn, cast

import sympy as sp
from typing_extensions import override

from pytools import memoize_method, single_valued

import sumpy.symbolic as sym
from sumpy.expansion import (
    ExpansionBase,
    LinearPDEConformingVolumeTaylorExpansion,
    VolumeTaylorExpansion,
    VolumeTaylorExpansionMixin,
)
from sumpy.expansion.multipole import H2DMultipoleExpansion, Y2DMultipoleExpansion
from sumpy.tools import add_to_sac, mi_increment_axis


if TYPE_CHECKING:
    from collections.abc import Sequence

    import loopy as lp

    from sumpy.assignment_collection import SymbolicAssignmentCollection
    from sumpy.derivative_taker import DerivativeCoeffDict
    from sumpy.expansion.diff_op import MultiIndex
    from sumpy.expansion.line_recurrence import _LineRecurrenceTable
    from sumpy.expansion.m2l import M2LTranslationBase, TranslationClassesDepData
    from sumpy.expansion.multipole import (
        HankelBased2DMultipoleExpansion,
        MultipoleExpansionBase,
    )
    from sumpy.kernel import ScalarKernel


logger = logging.getLogger(__name__)

__doc__ = """
.. autoclass:: LocalExpansionBase
.. autoclass:: VolumeTaylorLocalExpansion
.. autoclass:: H2DLocalExpansion
.. autoclass:: Y2DLocalExpansion
.. autoclass:: LineTaylorLocalExpansion
.. autoclass:: ODELineTaylorLocalExpansion
"""


@dataclass(frozen=True)
class LocalExpansionBase(ExpansionBase, ABC):
    """Base class for local expansions.

    .. automethod:: translate_from
    """

    @property
    @abstractmethod
    def m2l_translation(self) -> M2LTranslationBase:
        ...

    @abstractmethod
    def translate_from(self,
                src_expansion: LocalExpansionBase | MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
                m2l_translation_classes_dependent_data: (
                       TranslationClassesDepData | None) = None
            ) -> Sequence[sym.Expr]:
        """Translate from a multipole or local expansion to a local expansion

        :arg src_expansion: The source expansion to translate from.
        :arg src_coeff_exprs: An iterable of symbolic expressions representing the
                coefficients of the source expansion.
        :arg src_rscale: scaling factor for the source expansion.
        :arg dvec: symbolic expression for the distance between target and
                source centers.
        :arg tgt_rscale: scaling factor for the target expansion.
        :arg sac: An object of type
                :class:`sumpy.assignment_collection.SymbolicAssignmentCollection`
                to collect common subexpressions or None.
        :arg m2l_translation_classes_dependent_data: An iterable of symbolic
                expressions representing the expressions returned by
                :func:`~sumpy.expansion.m2l.M2LTranslationBase.translation_classes_dependent_data`.
        """


# {{{ line taylor

class LineTaylorLocalExpansion(LocalExpansionBase, ABC):
    @property
    @override
    def m2l_translation(self) -> M2LTranslationBase:
        # FIXME: Um...
        raise NotImplementedError()

    @override
    def get_storage_index(self, mi: MultiIndex) -> int:
        ind, = mi
        return ind

    @override
    def get_coefficient_identifiers(self) -> Sequence[MultiIndex]:
        return [(i,) for i in range(self.order+1)]

    @override
    def coefficients_from_source(self,
                kernel: ScalarKernel,
                avec: sym.Matrix,
                bvec: sym.Matrix | None,
                rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None
            ) -> Sequence[sym.Expr]:
        # no point in heeding rscale here--just ignore it
        if bvec is None:
            raise RuntimeError("cannot use line-Taylor expansions in a setting "
                    "where the center-target vector is not known at coefficient "
                    "formation")

        tau = sym.Symbol("tau")
        avec_line = cast("sym.Matrix", avec + tau*bvec)
        line_kernel = kernel.get_expression(avec_line)

        from sumpy.symbolic import USE_SYMENGINE

        if USE_SYMENGINE:
            from sumpy.derivative_taker import ExprDerivativeTaker
            deriv_taker = ExprDerivativeTaker(line_kernel, (tau,), sac=sac,
                                              rscale=sym.sympify(1))

            return [kernel.postprocess_at_source(deriv_taker.diff(i), avec)
                    .subs(tau, 0)
                    for i in self.get_coefficient_identifiers()]
        else:
            # Workaround for sympy. The automatic distribution after
            # single-variable diff makes the expressions very large
            # (https://github.com/sympy/sympy/issues/4596), so avoid doing
            # single variable diff.
            #
            # See also https://gitlab.tiker.net/inducer/pytential/merge_requests/12

            return [kernel.postprocess_at_source(line_kernel.diff(tau, i), avec)
                    .subs(tau, 0)
                    for i, in self.get_coefficient_identifiers()]

    @override
    def evaluate(self,
                 kernel: ScalarKernel,
                 coeffs: Sequence[sym.Expr],
                 bvec: sym.Matrix,
                 rscale: sym.Expr,
                 sac: SymbolicAssignmentCollection | None = None) -> sym.Expr:
        # no point in heeding rscale here--just ignore it

        # NOTE: We can't meaningfully apply target derivatives here.
        # Instead, this is handled in LayerPotentialBase._evaluate.
        return sym.Add(*(
                    coeffs[self.get_storage_index(i)] / math.factorial(i[0])
                    for i in self.get_coefficient_identifiers()))

    @override
    def translate_from(self,
                src_expansion: LocalExpansionBase | MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
                m2l_translation_classes_dependent_data: (
                       TranslationClassesDepData | None) = None
            ) -> Sequence[sym.Expr]:
        raise NotImplementedError


@dataclass(frozen=True)
class ODELineTaylorLocalExpansion(LocalExpansionBase):
    r"""Compute direct-QBX line Taylor expansions from the kernel's ODE.

    The base kernel is :math:`G(z) = g(q)`, with :math:`q = |z|^2`,
    and its PDE gives a polynomial-coefficient ODE for :math:`g`.
    Source and target kernels must share this base kernel.

    """

    target_kernels: Sequence[ScalarKernel] | None = field(kw_only=True, default=None)
    target_stored_identifiers: tuple[tuple[int, ...], ...] = field(init=False)

    @memoize_method
    def _target_kernel_derivatives(self, kernel: ScalarKernel) -> DerivativeCoeffDict:
        """
        Return ``{mi: weight}``.
        """
        from sumpy.derivative_taker import (
            DifferentiatedExprDerivativeTaker,
            ExprDerivativeTaker,
        )

        z = sym.make_sym_vector("z", self.dim)
        taker = DifferentiatedExprDerivativeTaker(
            ExprDerivativeTaker(sym.Integer(0), tuple(z)), {(0,)*self.dim: 1})
        return kernel.postprocess_at_target(taker, z).derivative_coeff_dict

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        target_stored_identifiers = ((0,)*self.dim,)
        if self.target_kernels is not None:
            object.__setattr__(self, "target_kernels", tuple(self.target_kernels))
            stored_identifiers = {}
            for target_kernel in self.target_kernels:
                stored_identifiers.update(dict.fromkeys(
                    sorted(self._target_kernel_derivatives(target_kernel))))
            target_stored_identifiers = tuple(stored_identifiers)
        object.__setattr__(self, "target_stored_identifiers", target_stored_identifiers)

    def _assign_states_to_sac(
            self, max_derivative_order: int, avec: sym.Matrix, bvec: sym.Matrix,
            sac: SymbolicAssignmentCollection,
    ) -> _LineRecurrenceTable[sp.Symbol]:
        r"""
        Assign :math:`U_{i,j} = [\tau^j]g^{(i)}(q(\tau))` in ``sac`` and return
        the table of assigned symbols.
        """
        # q(tau) = |avec + tau*bvec|^2 = q0 + q1*tau + q2*tau^2
        # Component-style names such as q2 are rewritten as vector accesses.
        q0 = sp.Symbol(sac.assign_unique("q0_line", avec.dot(avec)))
        q1 = sp.Symbol(sac.assign_unique("q1_line", 2*avec.dot(bvec)))
        q2 = sp.Symbol(sac.assign_unique("q2_line", bvec.dot(bvec)))

        from sumpy.expansion import line_recurrence
        from sumpy.expansion.kernel_ode import kernel_ode

        # kernel_ode returns the ODE coefficients and
        # initial conditions g^(i)(q0) for i = 0, ..., ode_order-1.
        ode_coeffs, init_cond_exprs = kernel_ode(self.kernel, q0)

        # If max_derivative_order >= ode_order, we need to differentiate the ODE
        # and extend the initial conditions to get g^(i)(q0)
        # for i <= max_derivative_order.
        init_cond_exprs, uij_update_exprs = line_recurrence.recurrence(
            ode_coeffs, max_derivative_order, init_conds=init_cond_exprs,
            q0=q0, q1=q1, q2=q2)

        assigned_init_conds = [
            sp.Symbol(sac.assign_unique(f"U{i}j0", value))
            for i, value in enumerate(init_cond_exprs)]

        # Example: 3D Laplace with M = 1, at i = 0 and j = 2. Starting from
        #     uij_update_exprs[0] = (q1*U[1, j-1] + 2*q2*U[1, j-2])/j,
        # set j = 2 and read the assigned states from the table:
        #     (q1*U1j1 + 2*q2*U1j0)/2.
        # Collect by state, then assign the coefficients and the new state:
        #     c0j2 = q2, c0j2_0 = q1/2,
        #     U0j2 = U1j0*c0j2 + U1j1*c0j2_0.
        def assign_uij_update(
                j: int, states: _LineRecurrenceTable[sp.Symbol],
        ) -> list[sp.Symbol]:
            r"""
            Return the assigned state symbols :math:`U_{i,j}`.
            """
            assigned_states = []
            for i, update_expr in enumerate(uij_update_exprs):
                update_expr = update_expr.xreplace({line_recurrence.j: j})
                state_substitutions = {}
                for state in update_expr.atoms(sp.Indexed):
                    state_i, state_j = state.indices
                    state_substitutions[state] = states[int(state_i), int(state_j)]
                update_expr = update_expr.xreplace(state_substitutions)

                state_coeffs = sp.collect(
                    sp.expand(update_expr), states.values(), evaluate=False)

                terms = []
                for state in sorted(state_coeffs, key=str):
                    name = sac.assign_unique(f"c{i}j{j}", state_coeffs[state])
                    terms.append(sp.Symbol(name)*state)
                name = sac.assign_unique(f"U{i}j{j}", sp.Add(*terms))
                assigned_states.append(sp.Symbol(name))
            return assigned_states

        from sumpy.expansion.line_recurrence import line_recurrence_table

        return line_recurrence_table(
            assigned_init_conds, assign_uij_update, expansion_order=self.order)

    def _line_chain_rule_weights(
            self, mi: tuple[int, ...], avec: sym.Matrix, bvec: sym.Matrix,
    ) -> dict[tuple[int, int], sp.Expr]:
        r"""
        Return the chain-rule weights :math:`\{(i,k): C_{i,k}\}` of
        :math:`(\partial_z^{\mathrm{mi}}G)(z)\big|_{z=z(\tau)}
        = \sum_{i,k} C_{i,k}\tau^k g^{(i)}(q(\tau))`, where
        :math:`z(\tau) = \mathrm{avec} + \tau\mathrm{bvec}` and
        :math:`q(\tau) = |z(\tau)|^2`.
        Only terms with :math:`k \le p` are returned, where :math:`p` is the
        expansion order.
        """
        from sumpy.expansion.kernel_ode import chain_rule_weights

        tau = sp.Symbol("tau")
        z = sp.Matrix(sp.symbols(f"z0:{self.dim}"))
        z_tau = dict(zip(z, avec + tau*bvec, strict=False))
        C = {}  # noqa: N806
        for i, weight in chain_rule_weights(mi, z).items():
            polynomial = sp.Poly(weight.xreplace(z_tau), tau)
            for (k,), cik in polynomial.terms():
                if k <= self.order:
                    C[i, k] = cik
        return C

    @override
    def coefficients_from_source_vec(
            self, kernels: Sequence[ScalarKernel],
            avec: sym.Matrix, bvec: sym.Matrix | None,
            rscale: sym.Expr, weights: Sequence[sym.Expr],
            sac: SymbolicAssignmentCollection | None = None,
    ) -> list[sp.Expr]:
        r"""
        :returns: a list of weighted line Taylor sums at :math:`\tau=1`,
            ordered by ``target_stored_identifiers``.
        """
        from sumpy.tools import add_mi

        # src_order is the highest |mi| across all source kernels.
        # tgt_order is the highest |mi| across all target kernels.
        src_order = 0
        for src_knl in kernels:
            src_knl_mis = src_knl.get_derivative_coeff_dict_at_source(
                {(0,)*self.dim: 1}).keys()
            src_order = max(src_order, max(sum(mi) for mi in src_knl_mis))
        tgt_order = max(sum(mi) for mi in self.target_stored_identifiers)

        # One shared U[i,j] table supplies all source/target kernels.
        uij_table = self._assign_states_to_sac(
            tgt_order+src_order, avec, bvec, sac)

        # Example: 3D Helmholtz S+D.
        # G(z) = g(q) = exp(I*k*sqrt(q))/sqrt(q), where q = |z|^2.
        # The ODE is q*g^(2) + (3/2)*g^(1) + (k^2/4)*g = 0.
        # Before global kernel scaling, the single and double layers are
        #     S(z) = g(|z|^2),
        #     D(z) = <-n, grad_z G(z)> = -2*<n, z>*g^(1)(|z|^2).
        #
        # Along z(tau) = avec + tau*bvec, with q(tau) = |z(tau)|^2,
        #     S(z(tau)) = g(q(tau)),
        #     D(z(tau)) = -2*<n, avec>*g^(1)(q(tau))
        #                 - 2*<n, bvec>*tau*g^(1)(q(tau)).
        #
        # With U[i, j] = [tau^j] g^(i)(q(tau)),
        #     [tau^j] S(z(tau)) = U[0, j],
        #     [tau^j] D(z(tau)) = -2*<n, avec>*U[1, j]
        #                         - 2*<n, bvec>*U[1, j-1].
        #
        # Sum through Taylor degree p and evaluate at tau = 1:
        #     S_p = sum_{j=0}^p U[0, j],
        #     D_p = -2*<n, avec>*sum_{j=0}^p U[1, j]
        #           -2*<n, bvec>*sum_{j=0}^{p-1} U[1, j].
        # The source contribution is strength_S*S_p + strength_D*D_p.

        # We define uij_partial_sums[i, j] = U[i, 0] + ... + U[i, j].
        # Then,
        #     S_p = uij_partial_sums[0, p],
        #     D_p = -2*<n, avec>*uij_partial_sums[1, p]
        #           -2*<n, bvec>*uij_partial_sums[1, p-1].

        uij_partial_sums = {}
        for (i, j), uij in uij_table.items():
            if j == 0:
                uij_partial_sums[i, 0] = uij

        taylor_sums = []
        for tgt_mi_idx, tgt_mi in enumerate(self.target_stored_identifiers):
            taylor_sum = 0
            for src_knl_idx, src_knl in enumerate(kernels):
                src_knl_mi_weights = src_knl.get_derivative_coeff_dict_at_source(
                    {(0,)*self.dim: 1}).items()

                C = defaultdict(int)  # noqa: N806
                for src_knl_mi, src_knl_weight in src_knl_mi_weights:
                    mi = add_mi(src_knl_mi, tgt_mi)
                    # \partial_z^{mi} G(z(tau))
                    #     = \sum_{i,k} cik*tau^k*g^(i)(q(tau))
                    #     = \sum_{i,k} cik*tau^k*(U[i,0] + U[i,1]*tau + ...).
                    #
                    # We only need
                    #   [tau^k] \partial_z^{mi} G(z(tau)) for k = 0, ..., self.order.
                    for (i, k), cik in self._line_chain_rule_weights(
                            mi, avec, bvec).items():
                        C[i, k] += src_knl_weight*cik

                src_knl_taylor_sum = 0
                for (i, k), cik in sorted(C.items()):
                    # For each (i, k), the contribution to src_knl_taylor_sum is:
                    #     cik * tau^k * (U[i,0] + U[i,1] + ... + U[i,self.order-k])
                    #    = cik * uij_partial_sums[i, self.order-k]
                    # where tau is evaluated at 1.
                    if cik == 0:
                        continue
                    assigned_cik = sp.Symbol(sac.assign_unique(
                        f"Ct{tgt_mi_idx}s{src_knl_idx}i{i}k{k}", cik))
                    for j in range(1, self.order-k+1):
                        if (i, j) not in uij_partial_sums:
                            # uij_partial_sums[i,j]
                            #     = uij_partial_sums[i,j-1]+U[i,j].
                            uij_partial_sums[i, j] = sp.Symbol(sac.assign_unique(
                                f"S{i}j{j}",
                                uij_partial_sums[i, j-1]+uij_table[i, j]))
                    src_knl_taylor_sum += (
                        assigned_cik*uij_partial_sums[i, self.order-k])
                taylor_sum += weights[src_knl_idx]*src_knl_taylor_sum
            taylor_sums.append(taylor_sum)
        return taylor_sums

    @override
    def coefficients_from_source(
            self, kernel: ScalarKernel, avec: sym.Matrix, bvec: sym.Matrix | None,
            rscale: sym.Expr, sac: SymbolicAssignmentCollection | None = None,
    ) -> list[sp.Expr]:
        return self.coefficients_from_source_vec(
            (kernel,), avec, bvec, rscale, (1,), sac=sac)

    @override
    def evaluate(
            self, kernel: ScalarKernel, coeffs: Sequence[sp.Expr], bvec: sym.Matrix,
            rscale: sym.Expr, sac: SymbolicAssignmentCollection | None = None,
    ) -> sp.Expr:
        r"""
        Return the Taylor expansion for the given target kernel and related source
        kernels at :math:`\tau=1`.

        For single- and double-layer source kernels :math:`S,D` with weights
        :math:`w_S,w_D` and target kernel :math:`G`, return
        :math:`w_S S_p+w_D D_p`, where :math:`S_p,D_p` are the line Taylor
        expansions of order :math:`p` around :math:`\tau=0`, evaluated at
        :math:`\tau=1`.
        """
        result = 0
        for mi, weight in self._target_kernel_derivatives(kernel).items():
            result += weight*coeffs[self.get_storage_index(mi)]
        return result

    @property
    @override
    def m2l_translation(self) -> NoReturn:
        raise NotImplementedError

    @override
    def translate_from(self, *args: object, **kwargs: object) -> NoReturn:
        raise NotImplementedError

    @override
    def get_coefficient_identifiers(self) -> list[tuple[int, ...]]:
        return list(self.target_stored_identifiers)

    @override
    def get_storage_index(self, mi: tuple[int, ...]) -> int:
        return self.target_stored_identifiers.index(mi)

# }}}


# {{{ volume taylor

@dataclass(frozen=True)
class VolumeTaylorLocalExpansionBase(VolumeTaylorExpansionMixin,
                                     LocalExpansionBase, ABC):
    """
    Coefficients represent derivative values of the kernel.
    """

    m2l_translation_override: M2LTranslationBase | None = \
        field(kw_only=True, default=None)

    @property
    @override
    def m2l_translation(self) -> M2LTranslationBase:
        if self.m2l_translation_override is not None:
            return self.m2l_translation_override
        else:
            from sumpy.expansion.m2l import DefaultM2LTranslationClassFactory
            factory = DefaultM2LTranslationClassFactory()
            return factory.get_m2l_translation_class(self.kernel, self.__class__)()

    @override
    def coefficients_from_source_vec(self,
                kernels: Sequence[ScalarKernel],
                avec: sym.Matrix,
                bvec: sym.Matrix | None,
                rscale: sym.Expr,
                weights: Sequence[sym.Expr],
                sac: SymbolicAssignmentCollection | None = None
            ) -> Sequence[sym.Expr]:
        """Form an expansion with a linear combination of kernels and weights.
        Since all of the kernels share a base kernel, this method uses one
        derivative taker with one SymbolicAssignmentCollection object
        to remove redundant calculations.

        :arg avec: vector from source to center.
        :arg bvec: vector from center to target. Not usually necessary,
            except for line-Taylor expansion.
        :arg sac: a symbolic assignment collection where temporary
            expressions are stored.

        :returns: a list of :mod:`sympy` expressions representing
            the coefficients of the expansion.
        """
        if not self.use_rscale:
            rscale = sym.sympify(1)

        base_kernel = single_valued(knl.get_base_kernel() for knl in kernels)
        base_taker = base_kernel.get_derivative_taker(avec, rscale, sac)
        result: list[sym.Expr] = [sym.sympify(0)]*len(self)

        for knl, weight in zip(kernels, weights, strict=True):
            taker = knl.postprocess_at_source(base_taker, avec)
            # Following is a hack to make sure cse works.
            if 1:
                def save_temp(x: sym.Expr) -> sym.Expr:
                    return add_to_sac(sac, weight * x)  # ruff:ignore[function-uses-loop-variable]

                for i, mi in enumerate(self.get_coefficient_identifiers()):
                    result[i] += taker.diff(mi, save_temp)
            else:
                def save_temp(x: sym.Expr) -> sym.Expr:
                    return add_to_sac(sac, x)

                for i, mi in enumerate(self.get_coefficient_identifiers()):
                    result[i] += weight * taker.diff(mi, save_temp)

        return result

    @override
    def coefficients_from_source(self,
                kernel: ScalarKernel,
                avec: sym.Matrix,
                bvec: sym.Matrix | None,
                rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None
            ) -> Sequence[sym.Expr]:
        return self.coefficients_from_source_vec((kernel,), avec, bvec,
                rscale=rscale, weights=(sym.sympify(1),), sac=sac)

    @override
    def evaluate(self,
                kernel: ScalarKernel,
                coeffs: Sequence[sym.Expr],
                bvec: sym.Matrix,
                rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
            ) -> sym.Expr:
        rscale = sym.sympify(1 if not self.use_rscale else rscale)

        evaluated_coeffs = (
            self.expansion_terms_wrangler.get_full_kernel_derivatives_from_stored(
                coeffs, rscale, sac=sac))

        bvec_scaled = [cast("sym.Expr", b*rscale**-1) for b in bvec]
        from sumpy.tools import mi_factorial, mi_power

        result = sym.sympify(sum(
            coeff
            * mi_power(bvec_scaled, mi, evaluate=False)
            / mi_factorial(mi)
            for coeff, mi in zip(
                    evaluated_coeffs, self.get_full_coefficient_identifiers(),
                    strict=True)))

        return kernel.postprocess_at_target(result, bvec)

    @override
    def translate_from(self,
                src_expansion: LocalExpansionBase | MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
                m2l_translation_classes_dependent_data: (
                       TranslationClassesDepData | None) = None,
                _fast_version: bool = True,
            ) -> Sequence[sym.Expr]:
        logger.info("building translation operator for %s: %s(%d) -> %s(%d): start",
                    src_expansion.kernel,
                    type(src_expansion).__name__,
                    src_expansion.order,
                    type(self).__name__,
                    self.order)

        if not self.use_rscale:
            src_rscale = sym.sympify(1)
            tgt_rscale = sym.sympify(1)

        from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansionBase

        # {{{ M2L

        if isinstance(src_expansion, VolumeTaylorMultipoleExpansionBase):
            m2l_result = self.m2l_translation.translate(self, src_expansion,
                src_coeff_exprs, src_rscale, dvec, tgt_rscale, sac,
                m2l_translation_classes_dependent_data)

            logger.info("building translation operator: done")
            return m2l_result

        # }}}

        assert isinstance(src_expansion, VolumeTaylorLocalExpansionBase)

        # {{{ L2L

        # not coming from a Taylor multipole: expand via derivatives
        # FIXME: this shouldn't need to be sympified, but many places still
        # pass in floats. removing it fails `test_m2m_and_l2l_exprs_simpler`
        rscale_ratio = add_to_sac(sac, sym.sympify(tgt_rscale/src_rscale))

        src_wrangler = src_expansion.expansion_terms_wrangler
        src_coeffs = (
            src_wrangler.get_full_kernel_derivatives_from_stored(
                src_coeff_exprs, src_rscale, sac=sac))

        src_mis = \
            src_expansion.expansion_terms_wrangler.get_full_coefficient_identifiers()
        src_mi_to_index = {mi: i for i, mi in enumerate(src_mis)}

        tgt_mis = self.expansion_terms_wrangler.get_coefficient_identifiers()
        tgt_mi_to_index = {mi: i for i, mi in enumerate(tgt_mis)}

        tgt_split = self.expansion_terms_wrangler._split_coeffs_into_hyperplanes()

        p = max(sum(mi) for mi in src_mis)
        result: list[sym.Expr] = [sym.sympify(0)] * len(tgt_mis)

        # Local expansion around the old center gives us that,
        #
        # $ u = \sum_{|q| \le p} (x-c_1)^q \frac{C_q}{q!} $
        #
        # where $c_1$ is the old center and $q$ is a multi-index,
        # $p$ is the order and $C_q$ is a coefficient of the local expansion around
        # the center $c_1$.
        #
        # Differentiating, we get,
        #
        # $ D_{r} u = \sum_{|q| \le p} \frac{C_{q}}{(q-r)!} (x - c_1)^{q - r}$.
        #
        # This algorithm uses the observation that L2L coefficients
        # have the following form in 2D
        #
        # $T_{m, n} = \sum_{i\le p-m, j\le p-n-m-i} C_{i+m, j+n}
        #                   d_x^i d_y^j \frac{1}{i! j!}$
        #
        # where $d$ is the distance between the centers and $T$ is the translated
        # coefficient. $T$ can be rewritten as follows.
        #
        # Let $Y1_{m, n} = \sum_{j\le p-m-n} C_{m, j+n} d_y^j \frac{1}{j!}$.
        #
        # Then, $T_{m, n} = \sum_{i\le p-m} Y1_{i+m, n} d_x^i \frac{1}{i!}$.
        #
        # Expanding this to 3D,
        # $T_{m, n, l} = \sum_{i \le p-m, j \le p-n-m-i, k \le p-n-m-l-i-j}
        #             C_{i+m, j+n, k+l} d_x^i d_y^j d_z^k \frac{1}{i! j! k!}$
        #
        # Let,
        # $Y1_{m, n, l} = \sum_{k\le p-m-n-l} C_{m, n, k+l} d_z^k \frac{1}{l!}$
        # and,
        # $Y2_{m, n, l} = \sum_{j\le p-m-n} Y1_{m, j+n, l} d_y^j \frac{1}{n!}$.
        #
        # Then,
        # $T_{m, n, l} = \sum_{i\le p-m} Y2_{i+m, n, l} d_x^i \frac{1}{m!}$.
        #
        # Cost of the above algorithm is $O(p^4)$ for full since each value needs
        # $O(p)$ work and there are $O(p^3)$ values for $T, Y1, Y2$.
        # For a hyperplane of coefficients with normal direction `l` fixed,
        # we need only $O(p^2)$ of $T, Y1, Y2$ and since there are only a constant
        # number of coefficient hyperplanes in compressed, the algorithm is
        # $O(p^3)$

        # We start by iterating through all the axes which is at most 3 iterations
        # (in <=3D).
        # The number of iterations is one for full because all the $O(p)$ hyperplanes
        # are parallel to each other.
        # The number of iterations is one for compressed expansions with
        # elliptic PDEs because the $O(1)$ hyperplanes are parallel to each other.
        for axis in {d for d, _ in tgt_split}:
            # Use the axis as the first dimension to vary so that the below
            # algorithm is O(p^{d+1}) for full and O(p^{d}) for compressed
            dims = [axis, *list(range(axis)), *list(range(axis + 1, self.dim))]

            # Start with source coefficients. Gets updated after each axis.
            cur_dim_input_coeffs = list(src_coeffs)
            cur_dim_output_coeffs: list[sym.Expr] = [sym.sympify(-1)] * len(src_mis)

            # O(1) iterations
            for d in dims:
                cur_dim_output_coeffs = [sym.sympify(0)] * len(src_mis)

                # Only O(p^{d-1}) operations are used in compressed
                # O(p^d) operations are used in full
                for out_i, out_mi in enumerate(src_mis):
                    # O(p) iterations
                    for q in range(p+1-sum(out_mi)):
                        src_mi = mi_increment_axis(out_mi, d, q)
                        if src_mi in src_mi_to_index:
                            dvec_d = cast("sym.Expr", dvec[d])

                            cur_dim_output_coeffs[out_i] += (
                                (dvec_d/src_rscale)**q
                                * cur_dim_input_coeffs[src_mi_to_index[src_mi]]
                                / math.factorial(q))

                # Y at the end of the iteration becomes the source coefficients
                # for the next iteration
                cur_dim_input_coeffs = cur_dim_output_coeffs

            for mi in tgt_mis:
                # In L2L, source level usually has same or higher order than target
                # level. If not, extra coeffs in target level are zero filled.
                if mi not in src_mi_to_index:
                    result[tgt_mi_to_index[mi]] = sym.sympify(0)
                else:
                    # Add to result after scaling
                    result[tgt_mi_to_index[mi]] += (
                        cur_dim_output_coeffs[src_mi_to_index[mi]]
                        * rscale_ratio ** sum(mi))

        # {{{ simpler, functionally equivalent code

        if not _fast_version:
            # Rscale/operand magnitude is fairly sensitive to the order of
            # operations--which is something we don't have fantastic control
            # over at the symbolic level. Scaling dvec, then differentiating,
            # and finally rescaling dvec leaves the expression needing a scaling
            # to compensate for differentiating which is done at the end.
            # This moves the two cancelling "rscales" closer to each other at
            # the end in the hope of helping rscale magnitude.
            from sumpy.derivative_taker import ExprDerivativeTaker

            dvec_scaled = sym.Matrix([d*src_rscale for d in dvec])
            expr = src_expansion.evaluate(src_expansion.kernel, src_coeff_exprs,
                        dvec_scaled, rscale=src_rscale, sac=sac)

            replace_dict = {d: d/src_rscale for d in dvec}
            taker = ExprDerivativeTaker(expr, dvec)
            rscale_ratio = sym.UnevaluatedExpr(tgt_rscale/src_rscale)

            result = [
                    (taker.diff(mi).xreplace(replace_dict) * rscale_ratio**sum(mi))
                    for mi in self.get_coefficient_identifiers()]

        # }}}

        logger.info("building translation operator: done")
        return result

    def loopy_translate_from(self, src_expansion: ExpansionBase) -> lp.TranslationUnit:
        from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansionBase

        if isinstance(src_expansion, VolumeTaylorMultipoleExpansionBase):
            return self.m2l_translation.loopy_translate(self, src_expansion)

        raise NotImplementedError(
            f"a direct loopy kernel for translation from "
            f"{src_expansion} to {self} is not implemented.")


class VolumeTaylorLocalExpansion(
        VolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):
    pass


class LinearPDEConformingVolumeTaylorLocalExpansion(
        LinearPDEConformingVolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):
    pass

# }}}


# {{{ 2D Bessel-based-expansion

@dataclass(frozen=True)
class FourierBesselLocalExpansionMixin(LocalExpansionBase, ABC):

    m2l_translation_override: M2LTranslationBase | None = (
            field(kw_only=True, default=None))

    @property
    @abstractmethod
    def mpole_expn_class(self) -> type[HankelBased2DMultipoleExpansion]:
        ...

    @property
    @override
    def m2l_translation(self) -> M2LTranslationBase:
        if self.m2l_translation_override is not None:
            return self.m2l_translation_override
        else:
            from sumpy.expansion.m2l import DefaultM2LTranslationClassFactory

            factory = DefaultM2LTranslationClassFactory()
            return factory.get_m2l_translation_class(self.kernel, self.__class__)()

    @abstractmethod
    def get_bessel_arg_scaling(self) -> sym.Expr:
        ...

    @override
    def get_storage_index(self, mi: MultiIndex) -> int:
        ind, = mi
        return self.order+ind

    @override
    def get_coefficient_identifiers(self) -> Sequence[MultiIndex]:
        return [(i,) for i in range(-self.order, self.order+1)]

    @override
    def coefficients_from_source(self,
                kernel: ScalarKernel,
                avec: sym.Matrix,
                bvec: sym.Matrix | None,
                rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None
            ) -> Sequence[sym.Expr]:
        if not self.use_rscale:
            rscale = sym.sympify(1)

        from sumpy.symbolic import Hankel1, sym_real_norm_2

        arg_scale = self.get_bessel_arg_scaling()

        # The coordinates are negated since avec points from source to center.
        source_angle_rel_center = sym.atan2(-avec[1], -avec[0])
        avec_len = sym_real_norm_2(avec)
        return [kernel.postprocess_at_source(
                    Hankel1(c, arg_scale * avec_len, 0)
                    * rscale ** abs(c)
                    * sym.exp(sym.I * c * source_angle_rel_center), avec)
                    for c, in self.get_coefficient_identifiers()]

    @override
    def evaluate(self,
                kernel: ScalarKernel,
                coeffs: Sequence[sym.Expr],
                bvec: sym.Matrix,
                rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
            ) -> sym.Expr:
        if not self.use_rscale:
            rscale = sym.sympify(1)

        from sumpy.symbolic import BesselJ, sym_real_norm_2
        bvec_len = sym_real_norm_2(bvec)
        target_angle_rel_center = sym.atan2(bvec[1], bvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        return sym.sympify(sum(coeffs[self.get_storage_index((c,))]
                   * kernel.postprocess_at_target(
                       BesselJ(c, arg_scale * bvec_len, 0)
                       / rscale ** abs(c)
                       * sym.exp(sym.I * c * -target_angle_rel_center), bvec)
                for c, in self.get_coefficient_identifiers()))

    @override
    def translate_from(self,
                src_expansion: LocalExpansionBase | MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
                m2l_translation_classes_dependent_data: (
                       TranslationClassesDepData | None) = None
            ) -> Sequence[sym.Expr]:
        from sumpy.symbolic import BesselJ, sym_real_norm_2

        if not self.use_rscale:
            src_rscale = sym.sympify(1)
            tgt_rscale = sym.sympify(1)

        arg_scale = self.get_bessel_arg_scaling()

        if isinstance(src_expansion, type(self)):
            dvec_len = sym_real_norm_2(dvec)
            new_center_angle_rel_old_center = sym.atan2(dvec[1], dvec[0])

            translated_coeffs: list[sym.Expr] = []
            for j, in self.get_coefficient_identifiers():
                translated_coeffs.append(
                    sum((src_coeff_exprs[src_expansion.get_storage_index((m,))]
                         * BesselJ(m - j, arg_scale * dvec_len, 0)
                         / src_rscale ** abs(m)
                         * tgt_rscale ** abs(j)
                         * sym.exp(sym.I * (m - j) * -new_center_angle_rel_old_center)
                    for m, in src_expansion.get_coefficient_identifiers()),
                    sym.sympify(0)))

            return translated_coeffs

        if isinstance(src_expansion, self.mpole_expn_class):
            return self.m2l_translation.translate(self, src_expansion,
                src_coeff_exprs, src_rscale, dvec, tgt_rscale, sac,
                m2l_translation_classes_dependent_data)

        raise RuntimeError(
            "do not know how to translate "
            f"{type(src_expansion).__name__} to {type(self).__name__}")

    def loopy_translate_from(self, src_expansion: ExpansionBase) -> lp.TranslationUnit:
        if isinstance(src_expansion, self.mpole_expn_class):
            return self.m2l_translation.loopy_translate(self, src_expansion)

        raise NotImplementedError(
            f"a direct loopy kernel for translation from "
            f"{src_expansion} to {self} is not implemented.")


class H2DLocalExpansion(FourierBesselLocalExpansionMixin):
    def __post_init__(self) -> None:
        from sumpy.kernel import HelmholtzKernel

        kernel = self.kernel.get_base_kernel()
        if not (isinstance(kernel, HelmholtzKernel) and kernel.dim == 2):
            raise TypeError(
                f"{type(self).__name__} can only be applied to 2D HelmholtzKernel: "
                f"{kernel!r}")

    @property
    @override
    def mpole_expn_class(self) -> type[HankelBased2DMultipoleExpansion]:
        return H2DMultipoleExpansion

    @override
    def get_bessel_arg_scaling(self) -> sym.Expr:
        from sumpy.kernel import HelmholtzKernel
        kernel = self.kernel.get_base_kernel()
        assert isinstance(kernel, HelmholtzKernel)

        return sym.SpatialConstant(kernel.helmholtz_k_name).as_sympy()


class Y2DLocalExpansion(FourierBesselLocalExpansionMixin):
    def __post_init__(self) -> None:
        from sumpy.kernel import YukawaKernel

        kernel = self.kernel.get_base_kernel()
        if not (isinstance(kernel, YukawaKernel) and kernel.dim == 2):
            raise TypeError(
                f"{type(self).__name__} can only be applied to 2D YukawaKernel: "
                f"{kernel!r}")

    @property
    @override
    def mpole_expn_class(self) -> type[HankelBased2DMultipoleExpansion]:
        return Y2DMultipoleExpansion

    @override
    def get_bessel_arg_scaling(self) -> sym.Expr:
        from sumpy.kernel import YukawaKernel
        kernel = self.kernel.get_base_kernel()
        assert isinstance(kernel, YukawaKernel)

        return sym.I * sym.SpatialConstant(kernel.yukawa_lambda_name).as_sympy()

# }}}

# vim: fdm=marker
