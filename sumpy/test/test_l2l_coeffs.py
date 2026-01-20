from __future__ import annotations


__copyright__ = """
Copyright (C) 2026 Shawn/Chaoqi Lin
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


import math
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.special as spsp
import sympy as sp

from arraycontext import (
    pytest_generate_tests_for_array_contexts,
)

import sumpy.toys as t
from sumpy.array_context import PytestPyOpenCLArrayContextFactory, _acf  # noqa: F401
from sumpy.expansion.local import (
    LinearPDEConformingVolumeTaylorLocalExpansion,
    VolumeTaylorLocalExpansion,
)
from sumpy.kernel import (
    BiharmonicKernel,
    HelmholtzKernel,
    Kernel,
    LaplaceKernel,
    YukawaKernel,
)
from sumpy.tools import build_matrix


if TYPE_CHECKING:
    from collections.abc import Mapping

    from arraycontext import ArrayContextFactory


pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


def to_scalar(val):
    """Convert symbolic or array value to scalar."""
    if hasattr(val, "evalf"):
        val = val.evalf()
    if hasattr(val, "item"):
        val = val.item()
    return complex(val)


class NumericMatVecOperator:
    """Wrapper for symbolic matrix-vector operator with numeric
    substitution."""

    def __init__(self, symbolic_op, repl_dict):
        self.symbolic_op = symbolic_op
        self.repl_dict = repl_dict
        self.shape = symbolic_op.shape

    def matvec(self, vec):
        result = self.symbolic_op.matvec(vec)
        numeric_result = []
        for expr in result:
            if hasattr(expr, "xreplace"):
                numeric_result.append(
                    complex(expr.xreplace(self.repl_dict).evalf())
                )
            else:
                numeric_result.append(complex(expr))
        return np.array(numeric_result)


def get_repl_dict(kernel, extra_kwargs):
    """Numeric substitution for symbolic kernel parameters."""
    repl_dict = {}
    if "lam" in extra_kwargs:
        repl_dict[sp.Symbol("lam")] = extra_kwargs["lam"]
    if "k" in extra_kwargs:
        repl_dict[sp.Symbol("k")] = extra_kwargs["k"]
    return repl_dict


@pytest.mark.parametrize("knl,extra_kwargs", [
    (LaplaceKernel(2), {}),
    (YukawaKernel(2), {"lam": 0.1}),
    (HelmholtzKernel(2), {"k": 0.5}),
    (BiharmonicKernel(2), {}),
])
def test_l2l_coefficient_differences(
            actx_factory: ArrayContextFactory,
            knl: Kernel,
            extra_kwargs: Mapping[str, float],
            verbose: bool = True,
        ):
    """
    Tests that the expression for the difference between compressed and uncompressed
    translation in the compressed-expansions paper matches implemented reality.
    """
    order = 7
    dim = 2
    repl_dict = get_repl_dict(knl, extra_kwargs)

    # Setup sources and centers
    source = np.array([[5.0], [5.0]])
    c1 = np.array([0.0, 0.0])
    c2 = c1 + np.array([-0.5, 1.0])
    strength = np.array([1.0])

    actx = actx_factory()
    toy_ctx = t.ToyContext(
        knl,
        local_expn_class=LinearPDEConformingVolumeTaylorLocalExpansion,
        extra_kernel_kwargs=extra_kwargs
    )
    toy_ctx_full = t.ToyContext(
        knl,
        local_expn_class=VolumeTaylorLocalExpansion,
        extra_kernel_kwargs=extra_kwargs
    )

    # Compute expansions
    p = t.PointSources(toy_ctx, source, weights=strength)
    p_full = t.PointSources(toy_ctx_full, source, weights=strength)

    p2l = t.local_expand(actx, p, c1, order=order, rscale=1.0)
    p2l2l = t.local_expand(actx, p2l, c2, order=order, rscale=1.0)
    p2l_full = t.local_expand(actx, p_full, c1, order=order, rscale=1.0)
    p2l2l_full = t.local_expand(actx, p2l_full, c2, order=order, rscale=1.0)

    # Build matrix M
    p2l2l_expn = LinearPDEConformingVolumeTaylorLocalExpansion(knl, order)
    wrangler = p2l2l_expn.expansion_terms_wrangler
    M_symbolic = wrangler.get_projection_matrix(rscale=1.0)  # noqa: N806
    numeric_op = NumericMatVecOperator(M_symbolic, repl_dict)
    M = build_matrix(numeric_op, dtype=np.complex128)  # noqa: N806

    # Get compressed coefficients
    mu_c_symbolic = wrangler.get_full_kernel_derivatives_from_stored(
        p2l2l.coeffs, rscale=1.0
    )
    mu_c = []
    for coeff in mu_c_symbolic:
        if hasattr(coeff, "xreplace"):
            mu_c.append(to_scalar(coeff.xreplace(repl_dict)))
        else:
            mu_c.append(to_scalar(coeff))

    # Get identifiers
    stored_identifiers = p2l2l_expn.get_coefficient_identifiers()
    full_identifiers = p2l2l_expn.get_full_coefficient_identifiers()
    lexpn = VolumeTaylorLocalExpansion(knl, order)
    lexpn_idx = lexpn.get_full_coefficient_identifiers()

    h = c2 - c1
    global_const = to_scalar(knl.get_global_scaling_const())

    if verbose:
        print(f'\n{"="*104}')
        print(f"L2L Verification: {type(knl).__name__} (order={order})")
        print(f'{"="*104}')
        print(f"c1 = {c1}, c2 = {c2}, h = {h}")
        print()
        print(f"{'i':>3s} | {'ν(i)':>15s} | {'|ν|':4s} | "  # noqa: RUF001
              f"{'formula':>31s} | {'direct':>31s} | {'abs err':>10s}")
        print("-" * 104)

    max_abs_error = 0.0

    for i, nu_i in enumerate(full_identifiers):
        i_card = sum(np.array(nu_i))

        # Compute error by formula
        error = 0.0 + 0.0j
        for k, nu_jk in enumerate(stored_identifiers):
            jk_card = sum(np.array(nu_jk))
            if jk_card >= i_card:
                continue

            start_idx = math.comb(order - i_card + dim, dim)
            end_idx = math.comb(order - jk_card + dim, dim)

            for q_idx in range(start_idx, end_idx):
                nu_q = full_identifiers[q_idx]
                nu_sum = tuple(map(sum, zip(nu_q, nu_jk, strict=True)))
                if nu_sum not in full_identifiers:
                    continue

                deriv_idx = full_identifiers.index(nu_sum)
                gamma_deriv = to_scalar(p2l_full.coeffs[deriv_idx])
                h_pow = np.prod(h ** np.array(nu_q))
                fact_nu_q = np.prod(spsp.factorial(nu_q))

                error += -M[i, k] * gamma_deriv * h_pow / fact_nu_q

        error /= np.prod(spsp.factorial(nu_i))
        error *= global_const

        # Compute direct difference
        true_i_idx = lexpn_idx.index(nu_i)
        mu_full = to_scalar(p2l2l_full.coeffs[true_i_idx])
        direct_diff = (mu_full - mu_c[i]) / np.prod(spsp.factorial(nu_i))
        direct_diff *= global_const

        # Compute errors
        abs_err = abs(error - direct_diff)
        max_abs_error = max(max_abs_error, abs_err)

        if verbose:
            print(f"{i:3d} | {nu_i!s:>15s} | {i_card:4d} | "
                  f"{error.real: .8e}{error.imag:+.8e}j | "
                  f"{direct_diff.real: .8e}{direct_diff.imag:+.8e}j | "
                  f"{abs_err:9.2e}")

    if verbose:
        print(f"\nMaximum absolute error: {max_abs_error:.2e}")

    assert max_abs_error < 1e-10, \
        f"{type(knl).__name__}: error {max_abs_error:.2e}"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
