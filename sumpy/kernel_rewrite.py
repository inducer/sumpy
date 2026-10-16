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
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np

from pytools import (
    generate_nonnegative_integer_tuples_summing_to_at_most as gnitstam,
)

import sumpy.symbolic as sym
from sumpy.kernel import KernelWrapper, ScalarKernel


if TYPE_CHECKING:
    from collections.abc import Sequence

    import optype.numpy as onp

    from pymbolic.typing import ArithmeticExpression

    from sumpy.expansion.diff_op import MultiIndex

logger = logging.getLogger(__name__)

__doc__ = """
.. autoexception:: RewriteFailedError
.. autoclass:: LinearOperatorRepresentation

.. autofunction:: rewrite_using_base_kernel
"""

# {{{ rewrite_using_base_kernel


class RewriteFailedError(Exception):
    """Exception raised when a rewrite is not possible."""


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
    .. autoattribute:: const
    .. autoattribute:: coeffs
    """

    target_kernel: ScalarKernel
    """The target kernel expressed in terms of :attr:`base_kernel`."""
    base_kernel: ScalarKernel
    """A base kernel used to express the target kernel."""

    mis: Sequence[MultiIndex]
    """Multi-indices for each non-zero term in the linear combination."""
    const: ArithmeticExpression
    """Additive constant in the linear combination. Note that this cannot be
    expressed with a multi-indenx, since ``(0, 0, 0)`` corresponds to the no
    derivative case.
    """
    coeffs: Sequence[ArithmeticExpression]
    """Constant coefficients in the linear combination."""

    def pretty(self) -> str:
        from sumpy.kernel import AxisTargetDerivative

        terms: list[str] = []
        if self.const != 0:
            terms.append(str(self.const))

        for mi, c in zip(self.mis, self.coeffs, strict=True):
            expr = self.base_kernel
            for d, n in enumerate(mi):
                for _ in range(n):
                    expr = AxisTargetDerivative(d, expr)

            terms.append(f"{c} * {expr}")

        return f"{self.target_kernel} = " + " + ".join(terms)


def _get_base_kernel_derivatives(
    kernel: ScalarKernel,
) -> tuple[ScalarKernel, tuple[tuple[int, int], ...]]:
    """Strip all axis derivatives from *kernel*.

    :returns: the base kernel of *kernel* and a tuple of ``(sign, axis)`` pairs
        for each derivative, where *sign* is ``+1`` for target derivatives and
        ``-1`` for source derivatives.
    """
    from sumpy.kernel import AxisSourceDerivative, AxisTargetDerivative, DerivativeBase

    derivatives: list[tuple[int, int]] = []
    while isinstance(kernel, DerivativeBase):
        if isinstance(kernel, AxisTargetDerivative):
            sign = 1
        elif isinstance(kernel, AxisSourceDerivative):
            sign = -1
        else:
            break

        derivatives.append((sign, kernel.axis))
        kernel = kernel.inner_kernel

    return kernel, tuple(derivatives)


def _add_derivatives_to_representation(
        op: LinearOperatorRepresentation,
        target_kernel: ScalarKernel,
        ds: tuple[tuple[int, int], ...]
    ) -> LinearOperatorRepresentation:
    if not ds:
        return op

    mi_shift = [0] * target_kernel.dim
    fac = 1

    for sign, axis in ds:
        mi_shift[axis] += 1
        fac *= sign

    from sumpy.tools import add_mi

    return LinearOperatorRepresentation(
        target_kernel=target_kernel,
        base_kernel=op.base_kernel,
        mis=tuple(add_mi(mi, mi_shift) for mi in op.mis),
        const=0,
        coeffs=tuple(c if fac == 1 else -c for c in op.coeffs),
    )


def rewrite_using_base_kernel(
        target_kernel: ScalarKernel,
        base_kernel: ScalarKernel,
        *,
        method: Literal["fourier", "lu"] | None = None,
        min_order: int | None = None,
        rtol: float = 1.0e-8,
        rng: np.random.Generator | None = None,
    ) -> LinearOperatorRepresentation:
    """Rewrite the *target_kernel* as a :class:`LinearOperatorRepresentation`
    of the *base_kernel*.

    :arg min_order: (lu) minimum order guess for the required derivatives of
        *base_kernel* to obtain *target_kernel*.
    :arg rtol: (lu) relative tolerance used to determine if the linear combination
        of the *base_kernel* matches the *target_kernel*.
    :arg rng: (lu) used to generate random points for the LU-based rewrite.

    :raises RewriteFailedError: if no rewrite of the *target_kernels* in terms
        of the *base_kernel* is possible.
    """
    if min_order is not None and min_order <= 0:
        raise ValueError(f"'min_order' must be positive: {min_order}")

    if not 0.0 < rtol <= 1.0:
        raise ValueError(f"'rtol' must be in (0, 1]: {rtol}")

    dim = base_kernel.dim
    if target_kernel.dim != dim:
        raise ValueError(
            f"kernel dimension mismatch: 'target_kernel' is {target_kernel.dim}d "
            f"and 'base_kernel' is {dim}d"
        )

    inner_target_kernel, derivatives = _get_base_kernel_derivatives(target_kernel)

    if isinstance(inner_target_kernel, KernelWrapper):
        raise TypeError(f"'target_kernel' cannot be wrapped: {type(target_kernel)}")

    if isinstance(base_kernel, KernelWrapper):
        raise TypeError(f"'base_kernel' cannot be wrapped: {type(base_kernel)}")

    try:
        _ = base_kernel.get_pde_system_kernel()
    except TypeError:
        pass
    else:
        raise ValueError(
            f"'base_kernel' cannot be part of a system: {type(base_kernel)}"
        )

    pde = base_kernel.get_pde_as_diff_op()
    if min_order is None:
        min_order = pde.order

    if method is None:
        try:
            result = rewrite_using_base_kernel_fourier(inner_target_kernel, base_kernel)
        except RewriteFailedError:
            result = rewrite_using_base_kernel_lu(
                inner_target_kernel, base_kernel,
                min_order=min_order,
                rtol=rtol,
                rng=rng,
            )
    elif method == "fourier":
        result = rewrite_using_base_kernel_fourier(inner_target_kernel, base_kernel)
    elif method == "lu":
        result = rewrite_using_base_kernel_lu(
            inner_target_kernel, base_kernel,
            min_order=min_order,
            rtol=rtol,
            rng=rng,
        )
    else:
        raise ValueError(f"unknown rewrite method: {method!r}")

    return _add_derivatives_to_representation(result, target_kernel, derivatives)


# }}}


# {{{ rewrite_using_base_kernel_fourier


def rewrite_using_base_kernel_fourier(
    target_kernel: ScalarKernel,
    base_kernel: ScalarKernel,
) -> LinearOperatorRepresentation:
    r"""Find a relation between *target_kernel* and *base_kernel* using the
    Fourier symbol of their respective PDEs.

    The algorithm works by computing the Fourier symbols :math:`P_{\text{base}}`
    and :math:`P_{\text{target}}` of the scalar PDEs satisfied by the two
    kernels. The target kernel can be expressed as a differential operator
    applied to the base kernel if and only if the ratio

    .. math::

        D(i \boldsymbol{k}) = \frac{P_{\text{base}}(\boldsymbol{k})}
                                   {P_{\text{target}}(\boldsymbol{k})}

    is a polynomial in :math:`\boldsymbol{k}`. When it is, each monomial
    :math:`\prod_j (i k_j)^{\alpha_j}` corresponds to the derivative
    :math:`\partial^{|\alpha|} / \partial \boldsymbol{r}^{\alpha}`.

    Source and target derivatives of the target kernel are accounted for by
    multiplying its symbol by the corresponding :math:`\pm i k_j` factors.
    """

    # NOTE: This is a no-symengine code path, even if the rest of sumpy
    # is being driven by symengine, because for now (2026-10) no polynomials.

    import sympy as sp

    dim = target_kernel.dim

    assert base_kernel.dim == dim
    assert not isinstance(target_kernel, KernelWrapper)
    assert not isinstance(base_kernel, KernelWrapper)

    from sumpy.expansion.diff_op import to_fourier_matrix

    # {{{ get Fourier symbols

    def to_sympy_matrix(mat: sym.Matrix) -> sp.Matrix:
        # NOTE: sumpy code uses the selected symbolic backend (sympy or symengine),
        # but the algebra below is done purely in sympy, so we have to convert
        return sp.Matrix([
            [sp.sympify(entry) for entry in row]
            for row in mat.tolist()
        ])

    # symbols passed to sumpy code use the backend, the sympy ones are for algebra
    ks = sym.make_sym_vector("_k", dim)
    pde_base = base_kernel.get_pde_as_diff_op()
    mat = to_sympy_matrix(to_fourier_matrix(pde_base, ks))
    fourier_base = 1 / mat[0, 0]

    try:
        target_system_kernel, idx = target_kernel.get_pde_system_kernel()
    except TypeError:
        target_system_kernel, idx = None, None

    if target_system_kernel is None:
        pde_target = target_kernel.get_pde_as_diff_op()
        mat = to_sympy_matrix(to_fourier_matrix(pde_target, ks))

        fourier_target = 1 / mat[0, 0]
    else:
        assert idx is not None

        try:
            pde_target_system = target_system_kernel.get_pde_as_diff_op()
        except NotImplementedError:
            raise RewriteFailedError(
                f"cannot rewrite kernel with unknown PDE: {target_system_kernel}"
            ) from None

        mat = to_sympy_matrix(to_fourier_matrix(pde_target_system, ks))

        # NOTE: this uses the cofactor formula to get the inverse of just (i, j)
        # NOTE: A 1x1 matrix
        i, j = idx
        fourier_target = sp.cancel(mat.cofactor(j, i) / mat.det())

    # }}}

    ks = sp.Matrix([sp.Symbol(f"_k{j}") for j in range(dim)])
    quotient = sp.simplify(fourier_target / fourier_base)
    if not quotient.is_polynomial(*ks):
        raise RewriteFailedError(
            f"cannot rewrite {target_kernel} in terms of {base_kernel}"
        )

    # {{{ construct linear operator

    mis: list[MultiIndex] = []
    coeffs: list[sp.Expr] = []

    for exponent, coeff in sp.Poly(quotient, *ks).as_dict().items():
        coeff = sp.simplify(coeff)
        if coeff == sp.Integer(0):
            continue

        # NOTE: a monomial c (i k)^alpha in the symbol corresponds to the
        # derivative c d/d r^alpha, so when reading off the coefficient of
        # k^alpha from the ratio, divide out the i^|alpha|
        mis.append(exponent)
        coeffs.append(sp.simplify(coeff / sp.I**sum(exponent)))

    if not mis:
        raise RewriteFailedError(
            f"cannot rewrite {target_kernel} in terms of {base_kernel}"
        )

    # }}}

    # {{{ fix scaling and additive constant

    dvec = sym.Matrix([sym.Symbol(f"d{i}") for i in range(dim)])
    base_kernel_expr = sp.sympify(
        base_kernel.get_global_scaling_const()
        * base_kernel.get_expression(dvec)
    )
    target_kernel_expr = sp.sympify(
        target_kernel.get_global_scaling_const()
        * target_kernel.get_expression(dvec)
    )

    # build the differential operator part in real space
    from sumpy.tools import mi_derivative
    dvec = sp.Matrix([sp.Symbol(f"d{i}") for i in range(dim)])

    base_kernel_expr_diff: sp.Expr = sp.Integer(0)
    for mi, c in zip(mis, coeffs, strict=True):
        base_kernel_expr_diff += c * mi_derivative(base_kernel_expr, dvec, mi)

    # NOTE: the coefficients obtained from `fourier_target / fourier_base` only
    # define the linear representation up to a factor and an additive constant.
    #
    # For example, taking Laplace and Biharmonic in 3D, we get that
    #   D(base_kernel_expr) = \Delta (-r / (8 * pi))
    #                       = -1 / (4 * pi * r)
    #                       = -target_kernel_expr
    #
    # We then also have the additive constant. We fix these by requiring that
    #   target_kernel_expr - fac * base_kernel_expr_diff
    # is a constant (independent of dvec). The loop below will almost always just
    # do one iteration, since `base_kernel_expr_diff` is not going to be constant.
    fac: sp.Expr | None = None
    for i in range(dim):
        di = sp.diff(base_kernel_expr_diff, dvec[i])
        if di != 0:
            fac = sp.simplify(sp.diff(target_kernel_expr, dvec[i]) / di)
            break

    if fac is None or (fac.free_symbols & set(dvec)):
        raise RewriteFailedError(
            f"cannot rewrite {target_kernel} in terms of {base_kernel}"
        )

    const = sp.simplify(target_kernel_expr - fac * base_kernel_expr_diff)
    if const.free_symbols & set(dvec):
        raise RewriteFailedError(
            f"cannot rewrite {target_kernel} in terms of {base_kernel}"
        )

    # }}}

    from pymbolic.mapper.flattener import flatten

    to_pymbolic = sym.SympyToPymbolicMapper(symbols=True)
    return LinearOperatorRepresentation(
        target_kernel,
        base_kernel,
        mis=mis,
        const=flatten(to_pymbolic(const)),
        coeffs=[flatten(to_pymbolic(fac * coeff)) for coeff in coeffs],
    )


# }}}


# {{{ rewrite_using_base_kernel_lu


class FactorizationFailedError(Exception):
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


def round_expr(
        expr: sym.Basic, atol: float = 1.0e-10, rtol: float = 1.0e-8
    ) -> sym.Basic:
    """Round all numeric values in *expr* to the nearest integer or fraction.

    This function clips all numbers close to zero (effectively removing them
    from the expression due to SymPy's automatic simplifications) and rounds all
    numbers to the nearest integer using the given *atol* and *rtol*.
    """
    from fractions import Fraction

    nums = expr.atoms(sym.Float)
    replace_dict: dict[Any, float] = {}

    for num in nums:
        value = float(num)
        nearest_int = round(value)

        if abs(value - nearest_int) < atol + rtol * abs(value):
            replace_dict[num] = sym.Integer(nearest_int)
        else:
            frac = Fraction(value).limit_denominator(1000)
            if abs(float(frac) - value) < atol + rtol * abs(value):
                replace_dict[num] = sym.Rational(frac.numerator, frac.denominator)
            else:
                replace_dict[num] = value

    return expr.xreplace(replace_dict)


def simplify(expr: sym.Basic) -> sym.Basic:
    if sym.USE_SYMENGINE:
        return expr.simplify()
    else:
        import sympy as sp

        return sp.simplify(expr)


def _generate_points_shells(
        dim: int,
        npoints: int, *,
        rmin: float = 0.25,
        rmax: float = 2.0,
        rng: np.random.Generator | None = None
    ) -> onp.Array2D[Any]:
    if dim < 1:
        raise ValueError(f"'dim' must be >= 1: {dim!r}")

    if npoints < 1:
        raise ValueError(f"'npoints' must be >= 1: {npoints!r}")

    if rmin >= rmax:
        raise ValueError(f"'rmin' must be smaller than 'rmax': {rmin} >= {rmax}")

    if rng is None:
        rng = np.random.default_rng()

    # make log spaced shell radii
    radii = np.exp(rng.uniform(np.log(rmin), np.log(rmax), npoints))

    # generate points on the unit sphere
    p = rng.standard_normal((dim, npoints))
    p /= np.linalg.norm(p, axis=0)

    return radii * p


def _make_expr_derivatives(
        base_expr: sym.Expr, dvec: sym.Matrix, mis: Sequence[MultiIndex]
    ) -> dict[MultiIndex, sym.Basic]:
    mi_to_derivative: dict[MultiIndex, sym.Basic] = {}
    for mi in mis:
        expr = base_expr
        for i, nderivs in enumerate(mi):
            if nderivs == 0:
                continue
            expr = expr.diff(dvec[i], nderivs)

        mi_to_derivative[mi] = expr

    return mi_to_derivative


def _make_derivative_matrix(
        points: onp.Array2D[Any],
        dvec: sym.Matrix,
        mis: Sequence[MultiIndex],
        mi_to_derivative: dict[MultiIndex, sym.Basic]
    ) -> sym.Matrix:
    # evaluate derivatives at points and construct matrix
    entries: list[list[sym.Basic]] = []
    for i in range(points.shape[1]):
        row: list[sym.Basic] = [sym.Integer(1)]

        for mi in mis:
            expr = mi_to_derivative[mi].xreplace(
                dict(zip(dvec, points[:, i], strict=True))
            )
            row.append(evalf(expr))

        entries.append(row)

    return sym.Matrix(entries)


def _check_linear_combination(
        target_expr: sym.Expr,
        base_expr: sym.Expr,
        dvec: sym.Matrix,
        mis: Sequence[MultiIndex],
        const: sym.Basic,
        coeffs: Sequence[sym.Basic],
        *,
        rng: np.random.Generator,
        rtol: float = 1.0e-8,
    ) -> bool:
    dim = len(dvec)
    mi_to_derivative = _make_expr_derivatives(base_expr, dvec, mis)
    points = _generate_points_shells(dim, len(mis) + 1, rng=rng)

    max_lhs = 0.0
    max_err = 0.0
    for i in range(points.shape[1]):
        subst = dict(zip(dvec, points[:, i], strict=True))
        lhs = evalf(target_expr.xreplace(subst))

        rhs = const
        for c, mi in zip(coeffs, mis, strict=True):
            rhs += evalf(c * mi_to_derivative[mi].xreplace(subst))

        if lhs.is_number and rhs.is_number:
            max_lhs = max(max_lhs, abs(lhs))
            max_err = max(max_err, abs(lhs - rhs))
        else:
            max_err = round_expr(simplify(lhs - rhs), rtol=rtol)
            if not max_err.is_zero:
                return False

    return max_err <= rtol * max(max_lhs, 1.0)


def _get_base_kernel_matrix_lu_factorization(
        base_kernel: ScalarKernel,
        order: int,
        *,
        rng: np.random.Generator,
        retries: int,
    ) -> _LUDecomposition:
    pde = base_kernel.get_pde_as_diff_op()
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
    mi_to_derivative = _make_expr_derivatives(base_expr, dvec, mis)

    # try to LU factorize on random points
    for _ in range(retries):
        points = _generate_points_shells(dim, len(mis) + 1, rng=rng)
        mat = _make_derivative_matrix(points, dvec, mis, mi_to_derivative)

        # NOTE: this can only happen if the points are somehow degenerate, e.g.
        # points[:, 0] == points[:, 1], so it shouldn't happen under normal
        # operating conditions?
        try:
            L, U, perm = mat.LUdecomposition()  # ruff: ignore[non-lowercase-variable-in-function]
        except RuntimeError:
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


def rewrite_using_base_kernel_lu(
        target_kernel: ScalarKernel,
        base_kernel: ScalarKernel,
        *,
        min_order: int | None = None,
        rtol: float = 1.0e-10,
        retries: int = 5,
        rng: np.random.Generator | None = None,
    ) -> LinearOperatorRepresentation:
    """Find a relation between the *target_kernel* and the *base_kernel* using
    a numerical LU-based algorithm.

    The algorithm samples the *base_kernel* and its derivatives at random
    points to get a matrix ``A``. It also samples the target kernel at the same
    points to get a vector ``b``. We then solve the system ``A c = b`` using
    an LU factorization of ``A``. The solution ``c`` is the vector of coefficients
    in the :class:`LinearOperatorRepresentation`.

    :arg retries: maximum number of retries for each order. If the LU decomposition
        fails due to a poor choice of random points, it is retried several times.
    """
    dim = target_kernel.dim

    assert base_kernel.dim == dim
    assert not isinstance(target_kernel, KernelWrapper)
    assert not isinstance(base_kernel, KernelWrapper)

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

    dvec = sym.make_sym_vector("d", dim)
    target_expr = target_kernel.get_expression(dvec)
    base_expr = base_kernel.get_expression(dvec)

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
            coeff = round_expr(evalf(coeff), rtol=rtol)
            if coeff == 0:
                continue

            if i == 0:
                const = coeff
                logger.debug("  %s", const)
            else:
                mis.append(lu.mis[i - 1])
                coeffs.append(coeff)
                logger.debug("  + %s*%s.diff%s", coeff, base_kernel, lu.mis[i - 1])

        if coeffs and _check_linear_combination(
            target_expr,
            base_expr,
            dvec,
            mis,
            const,
            coeffs,
            rtol=rtol,
            rng=rng,
        ):
            break

        coeffs = []
        order += 1

    if not coeffs:
        raise RewriteFailedError(
            f"could not express {target_kernel} in terms of {base_kernel}"
        )

    from pymbolic.mapper.flattener import flatten

    def to_pymbolic(expr: sym.Expr) -> sym.Expr:
        return flatten(sym.to_pymbolic(simplify(expr), symbols=True))

    return LinearOperatorRepresentation(
        target_kernel, base_kernel, mis,
        const=to_pymbolic(const * target_scaling),
        coeffs=tuple(to_pymbolic(target_scaling / base_scaling * c) for c in coeffs)
    )


# }}}
