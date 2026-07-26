from __future__ import annotations


__copyright__ = """
Copyright (C) 2012 Andreas Kloeckner
Copyright (C) 2020 Isuru Fernando
Copyright (C) 2026 Alexandru Fikl
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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from pytools import (
    generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
)

import sumpy.symbolic as sym


if TYPE_CHECKING:
    from collections.abc import Sequence

    import optype.numpy as onp

    from pymbolic.typing import ArithmeticExpression

    from sumpy.expansions.diff_op import MultiIndex
    from sumpy.kernel import Kernel

logger = logging.getLogger(__name__)


# {{{ rewrite_using_base_kernel

@dataclass(frozen=True)
class LinearOperatorRepresentation:
    r"""Expresses a target kernel as a linear operator acting on a base kernel.

    .. math::

        G(\boldsymbol{r}) = C + \sum_{|\boldsymbol{\alpha}| < p}
            c_{\boldsymbol{\alpha}}
            \frac{\partial^{|\boldsymbol{\alpha}|} G_0}
                 {\partial \boldsymbol{r}^{\boldsymbol{\alpha}}}

    .. autoattribute:: target_kernel
    .. autoattribute:: base_kernel
    .. autoattribute:: mis
    .. autoattribute:: coeffs
    """

    target_kernel: Kernel
    """The target kernel expressed in terms of :attr:`base_kernel`."""
    base_kernel: Kernel
    """A base kernel used to express the target kernel."""

    mis: Sequence[MultiIndex]
    """Multi-indices for each non-zero term in the linear combination."""
    coeffs: Sequence[ArithmeticExpression]
    """Constant coefficients in the linear combination. Note that the coefficients
    have length ``len(mis) + 1``, where the first element is always the constant term.
    """

    def pretty(self) -> str:
        from sumpy.kernel import AxisTargetDerivative

        terms = []
        if self.coeffs[0] != 0:
            terms.append(str(self.coeffs[0]))

        for mi, c in zip(self.mis, self.coeffs[1:], strict=True):
            expr = self.base_kernel
            for d, n in enumerate(mi):
                for _ in range(n):
                    expr = AxisTargetDerivative(d, expr)

            terms.append(f"{c} * {expr}")

        return f"{self.target_kernel} = " + " + ".join(terms)


def rewrite_using_base_kernel(
        target_kernel: Kernel,
        base_kernel: Kernel,
        *,
        order: int | None = None,
        atol: float = 1.0e-10,
    ) -> LinearOperatorRepresentation:
    pde = base_kernel.get_pde_as_diff_op()
    if order is None:
        order = pde.order

    # TODO: pick the best algorithm here
    return rewrite_using_base_kernel_lu(
        target_kernel, base_kernel,
        order=order,
    )

# }}}


# {{{ rewrite_using_base_kernel_lu

INT_MAX = 10 ** 15


class FactorizationFailedError(Exception):
    pass


class RewriteFailedError(Exception):
    pass


class _LUDecomposition(NamedTuple):
    L: sym.Matrix
    U: sym.Matrix
    permutation: Sequence[tuple[int, int]]

    mis: Sequence[MultiIndex]
    """The multi-indices for which the derivatives were computed. These correspond
    to rows in the matrix and should be used to recover the expansion of the
    target kernel in terms of the base kernel.
    """
    points: onp.Array2D[Any]
    """An array of shape ``(dim, npoints)`` of points where the base kernel was
    evaluated to compute the current LU factorization.
    """


def evalf(expr: sym.Expr, prec: int = 100) -> sym.Expr:
    """Evaluate an expression numerically using ``prec`` number of bits."""
    from sumpy.symbolic import USE_SYMENGINE

    if USE_SYMENGINE:
        return expr.n(prec=prec)
    else:
        import sympy
        dps = int(sympy.log(2**prec, 10))
        return expr.n(n=dps)


def rewrite_using_base_kernel_lu(
        target_kernel: Kernel,
        base_kernel: Kernel,
        *,
        min_order: int | None = None,
        retries: int = 5,
        rng: np.random.Generator | None = None,
    ) -> LinearOperatorRepresentation:
    """Find a relation between the *target_kernel* and the *base_kernel* using
    a numerical LU-based algorithm.

    The algorithm samples the *base_kernel* and its derivatives at random
    points to get a matrix ``A``. It also samples the target kernel at the same
    points to get a vector ``b`` and solving for the system ``A c = b`` using
    an LU factorization of ``A``. The solution ``c`` is the vector of coefficients
    in the linear combination :class:`DifferentialRepresentation`.

    :arg order: starting maximum derivative order to use when attempting the
        decomposition. By default, this will be the order of the PDE solved by
        *base_kernel*.
    :arg retries: maximum number of retries for each order. If the LU decomposition
        fails due to a poor choice of random points, it is retried several times.
    """

    pde = base_kernel.get_pde_as_diff_op()
    if min_order is None:
        min_order = pde.order

    if min_order > pde.order:
        raise NotImplementedError(
            "Rewriting when the base kernel's derivatives are linearly dependent "
            "is not implemented")

    if rng is None:
        rng = np.random.default_rng()

    coeffs: list[sym.Basic] = []
    mis: list[MultiIndex] = []

    dim = base_kernel.dim
    dvec = sym.make_sym_vector("d", dim)
    target_expr = target_kernel.get_expression(dvec)

    target_scaling = target_kernel.get_global_scaling_const()
    base_scaling = base_kernel.get_global_scaling_const()

    order = min_order
    while order <= pde.order:
        try:
            lu = _get_base_kernel_matrix_lu_factorization(
                base_kernel, order, rng=rng, retries=retries
            )
        except FactorizationFailedError as exc:
            if order == pde.order:
                raise RewriteFailedError(
                    f"failed to compute LU factorization for orders in "
                    f"[{min_order}, {pde.order}] for base kernel {base_kernel} "
                    f"after {retries} retries"
                ) from exc

            order += 1
            continue

        # evaluate right-hand side
        b = sym.Matrix([
            target_expr.xreplace(dict(zip(dvec, lu.points[:, i], strict=True)))
            for i in range(lu.points.shape[1])
        ])

        # solve
        all_coeffs = sym.solve_lu(lu.L, lu.U, lu.permutation, b,
                                  postprocess=lambda x: x.expand())

        # gather all non-zero coefficients from the result
        const = sym.Integer(0)
        coeffs = []
        mis = []
        for i, coeff in enumerate(all_coeffs):
            coeff = sym.chop(sym.simplify(coeff))
            if coeff == 0:
                continue

            if i == 0:
                const = sym.to_pymbolic(sym.simplify(coeff * target_scaling))
                logger.debug("  %s", coeff)
            else:
                mi = lu.mis[i - 1]
                coeff = sym.simplify(coeff * target_scaling / base_scaling)

                mis.append(mi)
                coeffs.append(sym.to_pymbolic(coeff))
                logger.debug("  + %s*%s.diff%s", coeff, base_kernel, mi)

        if coeffs:
            coeffs.insert(0, const)
            break

        order += 1

    if not coeffs:
        raise RewriteFailedError(
            f"could not express {target_kernel} in terms of {base_kernel}"
        )

    return LinearOperatorRepresentation(target_kernel, base_kernel, mis, coeffs)


def _get_base_kernel_matrix_lu_factorization(
        base_kernel: Kernel,
        order: int,
        *,
        rng: np.random.Generator,
        retries: int,
    ) -> _LUDecomposition:
    pde = base_kernel.get_pde_as_diff_op()
    if order > pde.order:
        raise NotImplementedError(
            "Rewriting when the base kernel's derivatives are linearly dependent "
            "is not implemented")

    dim = base_kernel.dim

    mis = list(gnitstam(order, dim))
    if order == pde.order:
        pde_mis = [ident.mi for eq in pde.eqs for ident in eq]
        pde_mis = [mi for mi in pde_mis if sum(mi) == order]
        mis.remove(pde_mis[-1])

        logger.debug("Removing %s to avoid linear dependent mis", pde_mis[-1])

    # get sympy expression for the base kernel
    dvec = sym.make_sym_vector("d", dim)
    base_expr = base_kernel.get_expression(dvec)

    # evaluate all the needed derivatives
    mi_to_derivative: dict[MultiIndex, sym.Basic] = {}
    for mi in mis:
        expr = base_expr
        for i, nderivs in enumerate(mi):
            if nderivs == 0:
                continue
            expr = expr.diff(dvec[i], nderivs)

        mi_to_derivative[mi] = sym.simplify(expr)

    # try to LU factorize on random points
    for _ in range(retries):
        # TODO: is it faster to generate numbers and then sympify them?
        points = np.empty((dim, len(mis) + 1), dtype=object)
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                points[i, j] = sym.Integer(rng.integers(1, INT_MAX)) / INT_MAX

        # evaluate derivatives at points and construct matrix
        entries: list[list[sym.Basic]] = []
        for i in range(points.shape[1]):
            row: list[sym.Basic] = [sym.Integer(1)]

            for mi in mis:
                expr = mi_to_derivative[mi].replace(
                    dict(zip(dvec, points[:, i], strict=True))
                )
                row.append(evalf(sym.simplify(expr)))

            entries.append(row)
        mat = sym.Matrix(entries)

        # TODO: LUdecomposition in symengine is not implemented for non-square matrices
        try:
            L, U, perm = mat.LUdecomposition()  # noqa: N806
        except RuntimeError:
            # NOTE: symengine seems to throw a SymEngineError -> RuntimeError when
            # it fails to do the LU factorization due to rank-deficiency
            continue
        else:
            # NOTE: and sympy seems to set the last row of U to 0
            if not sym.USE_SYMENGINE and all(expr == 0 for expr in U[-1, :]):
                continue

        return _LUDecomposition(L, U, perm, mis, points)

    raise FactorizationFailedError(
        f"failed to compute LU factorization to order {order} for {base_kernel} "
        f"after {retries} retries"
    )

# }}}
