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

from arraycontext import (
    pytest_generate_tests_for_array_contexts,
)

import sumpy.toys as t
from .coeff_test_tools import NumericMatVecOperator, get_repl_dict, to_scalar
from sumpy.array_context import PytestPyOpenCLArrayContextFactory, _acf  # noqa: F401
from sumpy.expansion.local import (
    LinearPDEConformingVolumeTaylorLocalExpansion,
)
from sumpy.expansion.multipole import (
    LinearPDEConformingVolumeTaylorMultipoleExpansion,
    VolumeTaylorMultipoleExpansion,
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


@pytest.mark.parametrize("knl,extra_kwargs", [
    (LaplaceKernel(2), {}),
    (YukawaKernel(2), {"lam": 0.1}),
    (HelmholtzKernel(2), {"k": 0.5}),
    (BiharmonicKernel(2), {}),
])
def test_m2m_coefficient_differences(
            actx_factory: ArrayContextFactory,
            knl: Kernel,
            extra_kwargs: Mapping[str, float],
            verbose: bool = True,
        ):
    """
    Compares two approaches:
    1. Compress coefficients -> embed -> M2M translate
    2. M2M translate with full coefficients

    Verifies the difference formula between these two approaches.
    """
    order = 7
    dim = 2
    repl_dict = get_repl_dict(knl, extra_kwargs)
    global_const = to_scalar(knl.get_global_scaling_const())

    # Set up source, centers, and target
    source = np.array([[0.0], [0.1]])
    strength = np.array([1.0])

    m_center1 = np.array([0.0, 0.0])
    offset_direction = np.array([-0.5, 0.25])
    c2_c1_dist = 0.1
    m_center2 = m_center1 + c2_c1_dist * offset_direction
    h = m_center2 - m_center1

    target = np.array([[2.0], [2.0]])

    if verbose:
        print(f"M2M Coefficient Verification for {type(knl).__name__}:")
        print(f"m_center1 = {m_center1}")
        print(f"m_center2 = {m_center2}")
        print(f"h = m_center2 - m_center1 = {h}")
        print()
        print(f"{'k':>3s} | {'ν(k)':>15s} | {'|ν(k)|':6s} | "  # noqa: RUF001
              f"{'difference by formula':>31s} | "
              f"{'difference by direct computation':>31s} | "
              f"{'abs err':>10s}")
        print("-" * 120)

    actx = actx_factory()

    toy_ctx_full = t.ToyContext(
        knl,
        mpole_expn_class=VolumeTaylorMultipoleExpansion,
        extra_kernel_kwargs=extra_kwargs
    )

    toy_ctx_local = t.ToyContext(
        knl,
        local_expn_class=LinearPDEConformingVolumeTaylorLocalExpansion,
        extra_kernel_kwargs=extra_kwargs
    )

    p_full = t.PointSources(toy_ctx_full, source, weights=strength)
    p2m_full = t.multipole_expand(actx, p_full, m_center1, order=order, rscale=1.0)

    p_local = t.PointSources(toy_ctx_local, m_center2.reshape(2, 1), weights=strength)
    p2l = t.local_expand(actx, p_local, target, order=order)

    mexpn = LinearPDEConformingVolumeTaylorMultipoleExpansion(knl, order)

    # Build matrix M
    wrangler = mexpn.expansion_terms_wrangler
    M_symbolic = wrangler.get_projection_matrix(rscale=1.0)  # noqa: N806
    numeric_op = NumericMatVecOperator(M_symbolic, repl_dict)
    M = build_matrix(numeric_op, dtype=np.complex128)  # noqa: N806
    coeffs_full = (M @ p2l.coeffs) * global_const

    # Get coefficient identifiers
    stored_identifiers = mexpn.get_coefficient_identifiers()
    full_identifiers = mexpn.get_full_coefficient_identifiers()
    is_stored = [mi in stored_identifiers for mi in full_identifiers]
    stored_indices = [i for i, st in enumerate(is_stored) if st]

    mexpn_full = VolumeTaylorMultipoleExpansion(knl, order)
    mexpn_full_idx = mexpn_full.get_full_coefficient_identifiers()

    max_abs_error = 0.0

    for k, nu_k in enumerate(full_identifiers):
        k_card = sum(np.array(nu_k))
        # assume all coefficient values are 1
        alpha_k = 1

        true_k_idx = mexpn_full_idx.index(nu_k)
        basis_full = np.zeros(len(mexpn_full_idx), dtype=np.complex128)
        basis_full[true_k_idx] = alpha_k
        p2m_full_k = p2m_full.with_coeffs(basis_full)

        # M^T @ alpha
        basis_cmp = np.zeros(M.shape[0], dtype=np.complex128)
        basis_cmp[stored_indices] = M[k, :] * alpha_k

        # Embed back into full basis
        basis_cmp_full = np.zeros(len(mexpn_full_idx), dtype=np.complex128)
        for i, nu_i in enumerate(full_identifiers):
            if basis_cmp[i] != 0:
                true_i_idx = mexpn_full_idx.index(nu_i)
                basis_cmp_full[true_i_idx] = basis_cmp[i]

        p2m_cmp_k = p2m_full.with_coeffs(basis_cmp_full)

        p2m2m_cmp = t.multipole_expand(
            actx, p2m_cmp_k, m_center2, order=order
        ).eval(actx, target)
        p2m2m_full = t.multipole_expand(
            actx, p2m_full_k, m_center2, order=order
        ).eval(actx, target)

        direct_diff = (p2m2m_cmp - p2m2m_full)[0]

        error = 0.0 + 0.0j
        for s, nu_js in enumerate(stored_identifiers):
            nu_js_card = sum(np.array(nu_js))
            inner_sum = 0.0 + 0.0j

            if nu_js_card <= k_card:
                start_idx = math.comb(order - k_card + dim, dim)
                end_idx = math.comb(order - nu_js_card + dim, dim)

                for idx in range(start_idx, end_idx):
                    nu_l = full_identifiers[idx]
                    nu_sum = tuple(a + b for a, b in zip(nu_l, nu_js, strict=True))

                    if nu_sum not in full_identifiers:
                        continue

                    derivative_idx = full_identifiers.index(nu_sum)
                    h_pow = np.prod(h ** np.array(nu_l))
                    fact_nu_l = np.prod(spsp.factorial(nu_l))

                    inner_sum += coeffs_full[derivative_idx] * h_pow / fact_nu_l

            error += inner_sum * M[k, s]

        abs_err = abs(error - direct_diff)
        max_abs_error = max(max_abs_error, abs_err)

        if verbose:
            print(f"{k:3d} | {nu_k!s:>15s} | {k_card:6d} | "
                  f"{error.real: .8e}{error.imag:+.8e}j | "
                  f"{direct_diff.real: .8e}{direct_diff.imag:+.8e}j | "
                  f"{abs_err:9.2e}")

    if verbose:
        print(f"\nMaximum absolute error: {max_abs_error:.2e}")

    assert max_abs_error < 1e-15, (
        f"{type(knl).__name__}: error {max_abs_error:.2e}"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
