__copyright__ = "Copyright (C) 2022 Isuru Fernando"

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

from typing import Sequence
import pymbolic
import pymbolic.primitives as prim
import loopy as lp
import numpy as np
from sumpy.expansion import ExpansionBase
from sumpy.kernel import Kernel
import sumpy.symbolic as sym
from sumpy.assignment_collection import SymbolicAssignmentCollection
from sumpy.tools import gather_loopy_arguments, gather_loopy_source_arguments
from math import prod, gcd

import logging
logger = logging.getLogger(__name__)


def make_e2p_loopy_kernel(
        expansion: ExpansionBase, kernels: Sequence[Kernel]) -> lp.TranslationUnit:
    """
    This is a helper function to create a loopy kernel for multipole/local
    evaluation. This function uses symbolic expressions given by the expansion class,
    converts them to pymbolic expressions and generates a loopy
    kernel. Note that the loopy kernel returned has lots of expressions in it and
    takes a long time. Therefore, this function should be used only as a fallback
    when there is no "loop-y" kernel to evaluate the expansion.
    """
    dim = expansion.dim

    bvec = sym.make_sym_vector("b", dim)
    ncoeffs = len(expansion.get_coefficient_identifiers())

    rscale = sym.Symbol("rscale")

    sac = SymbolicAssignmentCollection()

    domains = [
        "{[idim]: 0<=idim<dim}",
        "{[iknl]: 0<=iknl<nresults}",
    ]
    insns = []
    insns.append(
        lp.Assignment(
            assignee="b[idim]",
            expression="target[idim]-center[idim]",
            temp_var_type=lp.Optional(None),
        ))
    target_args = gather_loopy_arguments((expansion,) + tuple(kernels))

    coeff_exprs = sym.make_sym_vector("coeffs", ncoeffs)
    coeff_names = [
        sac.add_assignment(f"result{i}",
            expansion.evaluate(knl, coeff_exprs, bvec, rscale, sac=sac))
        for i, knl in enumerate(kernels)]

    sac.run_global_cse()

    code_transformers = [expansion.get_code_transformer()] \
        + [kernel.get_code_transformer() for kernel in kernels]

    from sumpy.codegen import to_loopy_insns
    insns += to_loopy_insns(
            sac.assignments.items(),
            vector_names={"b", "coeffs"},
            pymbolic_expr_maps=code_transformers,
            retain_names=coeff_names,
            complex_dtype=np.complex128  # FIXME
            )

    result = pymbolic.var("result")

    # change result{i} = expr into result[i] += expr
    for i in range(len(insns)):
        insn = insns[i]
        if isinstance(insn, lp.Assignment) and \
                isinstance(insn.assignee, pymbolic.var) and \
                insn.assignee.name.startswith(result.name):
            idx = int(insn.assignee.name[len(result.name):])
            insns[i] = lp.Assignment(
                assignee=result[idx],
                expression=result[idx] + insn.expression,
                id=f"result_{idx}",
                depends_on=insn.depends_on,
            )

    loopy_knl = lp.make_function(domains, insns,
            kernel_data=[
                lp.GlobalArg("result", shape=(len(kernels),), is_input=True,
                    is_output=True),
                lp.GlobalArg("coeffs",
                    shape=(ncoeffs,), is_input=True, is_output=False),
                lp.GlobalArg("center, target",
                    shape=(dim,), is_input=True, is_output=False),
                lp.ValueArg("rscale", is_input=True),
                lp.ValueArg("itgt", is_input=True),
                lp.ValueArg("ntargets", is_input=True),
                lp.GlobalArg("targets",
                    shape=(dim, "ntargets"), is_input=True, is_output=False),
                *target_args,
                ...],
            name="e2p",
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
            fixed_parameters={"dim": dim, "nresults": len(kernels)},
            )

    loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
    for kernel in kernels:
        loopy_knl = kernel.prepare_loopy_kernel(loopy_knl)

    optimizations = []

    return loopy_knl, optimizations


def make_p2e_loopy_kernel(
        expansion: ExpansionBase, kernels: Sequence[Kernel],
        strength_usage: Sequence[int], nstrengths: int) -> lp.TranslationUnit:
    """
    This is a helper function to create a loopy kernel for multipole/local
    expression. This function uses symbolic expressions given by the expansion
    class, converts them to pymbolic expressions and generates a loopy
    kernel. Note that the loopy kernel returned has lots of expressions in it and
    takes a long time. Therefore, this function should be used only as a fallback
    when there is no "loop-y" kernel to evaluate the expansion.
    """
    dim = expansion.dim

    avec = sym.make_sym_vector("a", dim)
    ncoeffs = len(expansion.get_coefficient_identifiers())

    rscale = sym.Symbol("rscale")

    sac = SymbolicAssignmentCollection()

    domains = [
        "{[idim]: 0<=idim<dim}",
    ]
    insns = []
    insns.append(
        lp.Assignment(
            assignee="a[idim]",
            expression="center[idim]-source[idim]",
            temp_var_type=lp.Optional(None),
        ))
    source_args = gather_loopy_source_arguments((expansion,) + tuple(kernels))

    all_strengths = sym.make_sym_vector("strength", nstrengths)
    strengths = [all_strengths[i] for i in strength_usage]
    coeffs = expansion.coefficients_from_source_vec(kernels,
        avec, None, rscale, strengths, sac=sac)

    coeff_names = [
        sac.add_assignment(f"coeffs{i}", coeff) for i, coeff in enumerate(coeffs)
    ]

    sac.run_global_cse()

    code_transformers = [expansion.get_code_transformer()] \
        + [kernel.get_code_transformer() for kernel in kernels]

    from sumpy.codegen import to_loopy_insns
    insns += to_loopy_insns(
            sac.assignments.items(),
            vector_names={"a", "strength"},
            pymbolic_expr_maps=code_transformers,
            retain_names=coeff_names,
            complex_dtype=np.complex128  # FIXME
            )

    coeffs = pymbolic.var("coeffs")

    # change coeff{i} = expr into coeff[i] += expr
    for i in range(len(insns)):
        insn = insns[i]
        if isinstance(insn, lp.Assignment) and \
                isinstance(insn.assignee, pymbolic.var) and \
                insn.assignee.name.startswith(coeffs.name):
            idx = int(insn.assignee.name[len(coeffs.name):])
            insns[i] = lp.Assignment(
                assignee=coeffs[idx],
                expression=coeffs[idx] + insn.expression,
                id=f"coeff_{idx}",
                depends_on=insn.depends_on,
            )

    loopy_knl = lp.make_function(domains, insns,
            kernel_data=[
                lp.GlobalArg("coeffs",
                    shape=(ncoeffs,), is_input=True, is_output=True),
                lp.GlobalArg("center, source",
                    shape=(dim,), is_input=True, is_output=False),
                lp.GlobalArg("strength",
                    shape=(nstrengths,), is_input=True, is_output=False),
                lp.ValueArg("rscale", is_input=True),
                lp.ValueArg("isrc", is_input=True),
                lp.ValueArg("nsources", is_input=True),
                lp.GlobalArg("sources",
                    shape=(dim, "nsources"), is_input=True, is_output=False),
                *source_args,
                ...],
            name="p2e",
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
            fixed_parameters={"dim": dim},
            )

    loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")
    for kernel in kernels:
        loopy_knl = kernel.prepare_loopy_kernel(loopy_knl)

    return loopy_knl


def make_l2p_loopy_kernel_for_volume_taylor(expansion, kernels):
    dim = expansion.dim
    order = expansion.order
    ncoeffs = len(expansion)

    code_transformers = [expansion.get_code_transformer()] \
        + [kernel.get_code_transformer() for kernel in kernels]
    pymbolic_conv = sym.SympyToPymbolicMapper()

    max_deriv_order = 0
    sym_expr_dicts = []
    for kernel in kernels:
        expr_dict = {(0,)*dim: 1}
        expr_dict = kernel.get_derivative_coeff_dict_at_target(expr_dict)
        max_deriv_order = max(max_deriv_order, max(sum(mi) for mi in expr_dict))
        sym_expr_dict = {}
        for mi, coeff in expr_dict.items():
            coeff = pymbolic_conv(coeff)
            for transform in code_transformers:
                coeff = transform(coeff)
            sym_expr_dict[mi] = coeff
        sym_expr_dicts.append(sym_expr_dict)

    domains = [
        "{[idim]: 0<=idim<dim}",
        "{[iorder0]: 0<iorder0<=order}",
        "{[zero_idx]: 0<=zero_idx<max_deriv_order}",
        "{[icoeff]: 0<=icoeff<ncoeffs}",
    ]

    powers = pymbolic.var("power_b")
    iorder = pymbolic.var("iorder0")
    idim = pymbolic.var("idim")
    result = pymbolic.var("result")
    b = pymbolic.var("b")
    center = pymbolic.var("center")
    target = pymbolic.var("target")
    rscale = pymbolic.var("rscale")
    coeffs = pymbolic.var("coeffs")
    icoeff = pymbolic.var("icoeff")
    zero_idx = pymbolic.var("zero_idx")
    temporary_variables = []

    insns = [
        lp.Assignment(
            assignee=b[idim],
            expression=(target[idim] - center[idim])*(1/rscale),
            id="set_b",
            temp_var_type=lp.Optional(None),
        ),
        # We need negative index access in the array to be zero
        # However loopy does not support negative indices, and we
        # have an offset of max_deriv_order for array access and
        # the first max_deriv_order values are set to zero.
        lp.Assignment(
            assignee=powers[idim, zero_idx],
            expression=0,
            id="zero_monomials",
            temp_var_type=lp.Optional(None),
        ),
        lp.Assignment(
            assignee=powers[idim, max_deriv_order],
            expression=1,
            id="init_monomials",
            depends_on=frozenset(["zero_monomials"]),
        ),
        lp.Assignment(
            assignee=powers[idim, max_deriv_order + iorder],
            expression=(
                powers[idim, max_deriv_order + iorder - 1]*b[idim]*(1/iorder)),
            id="update_monomials",
            depends_on=frozenset(["set_b", "init_monomials"]),
        ),
    ]

    optimizations = [lambda knl: lp.tag_inames(knl, "e2p_iorder0:unr")]
    iorder = pymbolic.var("iorder1")
    wrangler = expansion.expansion_terms_wrangler

    from sumpy.expansion import LinearPDEConformingVolumeTaylorExpansion
    if not isinstance(expansion, LinearPDEConformingVolumeTaylorExpansion):
        v = [pymbolic.var(f"x{i}") for i in range(dim)]
        domains += ["{[iorder1]: 0<=1iorder1<=order}"]
        upper_bound = "iorder1"
        for i in range(dim - 1, 0, -1):
            domains += [f"{{ [{v[i]}]: 0<={v[i]}<={upper_bound} }}"]
            upper_bound += f"-{v[i]}"
        domains += [f"{{ [{v[0]}]: {upper_bound}<={v[0]}<={upper_bound} }}"]
        idx = wrangler.get_storage_index(v, iorder)

        for ikernel, expr_dict in enumerate(sym_expr_dicts):
            expr = sum(coeff * prod(powers[i,
                v[i] + max_deriv_order - mi[i]] for i in range(dim))
                * (1 / rscale ** sum(mi))
                for mi, coeff in expr_dict.items())

            insn = lp.Assignment(
                assignee=result[ikernel],
                expression=(result[ikernel]
                    + coeffs[idx] * expr),
                id=f"write_{ikernel}",
                depends_on=frozenset(["update_monomials"]),
            )
            insns.append(insn)
        optimizations.append(lambda knl: lp.tag_inames(knl, "e2p_iorder1:unr"))
    else:
        coeffs_copy = pymbolic.var("coeffs_copy")
        insns.append(lp.Assignment(
            assignee=coeffs_copy[0, icoeff],
            expression=coeffs[icoeff],
            id="copy_coeffs",
        ))
        # We need two rows for coeffs_copy since we cannot use inplace
        # updates due to parallel updates so we alternatively use
        # coeffs_copy[0, :] and coeffs_copy[1, :] to write and read from.
        temporary_variables.append(lp.TemporaryVariable(
            name="coeffs_copy",
            shape=(2, ncoeffs),
        ))
        base_kernel = kernels[0].get_base_kernel()
        deriv_id_to_coeff, = base_kernel.get_pde_as_diff_op().eqs

        ordering_key, axis_permutation = \
                wrangler._get_mi_ordering_key_and_axis_permutation()
        max_deriv_id = max(deriv_id_to_coeff, key=ordering_key)
        max_mi = max_deriv_id.mi

        if all(m != 0 for m in max_mi):
            raise NotImplementedError("non-elliptic PDEs")

        slowest_axis = axis_permutation[0]
        c = max_mi[slowest_axis]
        v = [pymbolic.var(f"x{i}") for i in range(dim)]
        v[slowest_axis], v[0] = v[0], v[slowest_axis]
        x0 = v[0]

        # sync_split is the maximum number of iterations in v[0] that we can do
        # before a synchronization is needed. For Laplace 2D there are two rows
        # of stored coeffs, and both of them can be calculated before a sync
        # is needed. For biharmonic 2D there are four rows in stored coeffs,
        # but synchronization needs to happen every two rows because calculating
        # the 6th row needs the 4th row synchronized
        sync_split = gcd(*[c - deriv_id.mi[slowest_axis]
                         for deriv_id in deriv_id_to_coeff])

        def get_domains(v, iorder, with_sync):
            domains = [f"{{ [{x0}_outer]: 0<={x0}_outer<={order//c} }}"]
            if with_sync:
                expr = f"{c//sync_split}*{x0}_sync_outer + {c}*{x0}_outer"
                domains += [f"{{ [{x0}_sync_outer]: 0<={expr}<={order} "
                    f"and 0<={x0}_sync_outer<{c//sync_split} }}"]
                expr += f" + {v[0]}_inner"
                domains += [f"{{ [{v[0]}_inner]: 0<={expr}<={order} "
                    f"and 0<={v[0]}_inner<{sync_split} }}"]
            else:
                expr = f"{v[0]}_inner + {c}*{x0}_outer"
                domains += [f"{{ [{v[0]}_inner]: 0<={expr}<={order} "
                    f"and 0<={v[0]}_inner<{c} }}"]
            domains += [f"{{ [{v[0]}]: {expr}<={v[0]}<={expr} }}"]
            domains += [f"{{ [{iorder}]: {v[0]}<={iorder}<={order} }}"]
            upper_bound = f"{iorder}-{v[0]}"
            for i in range(dim - 1, 1, -1):
                domains += [f"{{ [{v[i]}]: 0<={v[i]}<={upper_bound} }}"]
                upper_bound += f"-{v[i]}"
            domains += [
                f"{{ [{v[1]}]: {upper_bound}<={v[1]}<={upper_bound} }}"]
            return domains

        def get_idx(v):
            idx_sym = list(v)
            idx_sym[0] = v[0] % c
            idx = wrangler.get_storage_index(idx_sym)
            return idx

        domains += get_domains(v, iorder, with_sync=True)
        idx = get_idx(v)

        if c == sync_split:
            # We do not need to sync within the c rows.
            # Update the values from the c rows set coeffs_copy[p, :] from
            # the previous c rows set coeffs_copy[p-1, :]
            # and then read from coeffs_copy[p, :].
            # This code path is different to avoid an extra copy and
            # a synchronization step.
            prev_copy_idx = (v[0]//c - 1) % 2
            curr_copy_idx = (v[0]//c) % 2
        else:
            # We need to sync within the c rows.
            # Using the biharmonic 2D example:
            # - Update the rows 4, 5 at coeffs_copy[1, :] from values at
            #     coeffs_copy[0, :]
            # - Synchronize
            # - Copy the rows 4, 5 from coeffs_copy[1, :] to coeffs_copy[0, :]
            # - Synchronize
            # - Update the rows 6, 7 at coeffs_copy[1, :] from values at
            #     coeffs_copy[0, :]
            # - Synchronize
            # - Copy the rows 6, 7 from coeffs_copy[1, :] to coeffs_copy[0, :]
            # - Synchronize
            # - Read the rows 4, 5, 6, 7 from coeffs_copy[0, :]
            prev_copy_idx = 0
            curr_copy_idx = 1

        max_mi_sym = [v[i] - max_mi[i] for i in range(dim)]
        scale = -1/deriv_id_to_coeff[max_deriv_id]
        expr = 0
        for deriv_id, pde_coeff in deriv_id_to_coeff.items():
            if deriv_id == max_deriv_id:
                continue
            mi_sym = [max_mi_sym[i] + deriv_id.mi[i] for i in range(dim)]
            mi_sym[0] = mi_sym[0] % c
            expr += (coeffs_copy[prev_copy_idx,
                wrangler.get_storage_index(mi_sym)]
                     * (rscale**(sum(max_mi) - sum(deriv_id.mi))
                     * pymbolic_conv(pde_coeff) * scale))

        insns.append(lp.Assignment(
            assignee=coeffs_copy[curr_copy_idx, idx],
            expression=expr,
            id="update_coeffs",
            depends_on=frozenset(["copy_coeffs"]),
            depends_on_is_final=True,
            predicates=frozenset([prim.Comparison(v[0], ">=", c)]),
        ))

        if c != sync_split:
            # We now copy before synchronization
            v = [pymbolic.var(f"z{i}") for i in range(dim)]
            v[slowest_axis], v[0] = v[0], v[slowest_axis]
            iorder = pymbolic.var("iorder3")
            idx = get_idx(v)
            domains += get_domains(v, iorder, with_sync=True)[2:]

            insns.append(lp.Assignment(
                assignee=coeffs_copy[0, idx],
                expression=coeffs_copy[1, idx],
                id="copy_sync",
                depends_on=frozenset(["update_coeffs"]),
                depends_on_is_final=True,
                predicates=frozenset([prim.Comparison(v[0], ">=", c)]),
            ))

        v = [pymbolic.var(f"y{i}") for i in range(dim)]
        v[slowest_axis], v[0] = v[0], v[slowest_axis]
        iorder = pymbolic.var("iorder2")
        idx = get_idx(v)
        domains += get_domains(v, iorder, with_sync=False)[1:]

        if c == sync_split:
            # We did not have to sync within the c rows.
            # We last wrote to coeffs_copy[v[0]//c % 2, :] and we read from it.
            fetch_idx = (v[0]//c) % 2
        else:
            # We need to sync within the c rows.
            # We last wrote to coeffs_copy[0, :] and we read from it.
            fetch_idx = 0

        for ikernel, expr_dict in enumerate(sym_expr_dicts):
            expr = sum(coeff * prod(powers[i,
                v[i] + max_deriv_order - mi[i]] for i in range(dim))
                * (1 / rscale ** sum(mi))
                for mi, coeff in expr_dict.items())

            insn = lp.Assignment(
                assignee=result[ikernel],
                expression=(result[ikernel]
                    + coeffs_copy[fetch_idx, idx] * expr),
                id=f"write_{ikernel}",
                depends_on=frozenset(["update_monomials",
                    "update_coeffs" if c == sync_split else "copy_sync"]),
                depends_on_is_final=True,
            )
            insns.append(insn)

        tags = {
            "e2p_iorder1": "l.0",
            f"e2p_{x0}_outer": "unr",
            f"e2p_{x0}_inner": "unr",
            f"e2p_{v[0]}_inner": "unr",
            "e2p_iorder2": "unr",
        }
        if c != sync_split:
            tags["e2p_iorder3"] = "l.0"

        optimizations += [
            lambda knl: lp.tag_inames(knl, tags),
            lambda knl: lp.set_temporary_address_space(knl, "e2p_coeffs_copy",
                lp.AddressSpace.LOCAL),
            lambda knl: lp.split_iname(knl, "e2p_icoeff", 32, inner_tag="l.0"),
        ]

    target_args = gather_loopy_arguments((expansion,) + tuple(kernels))
    loopy_knl = lp.make_function(domains, insns,
            kernel_data=[
                lp.GlobalArg("result", shape=(len(kernels),), is_input=True,
                    is_output=True),
                lp.GlobalArg("coeffs",
                    shape=(ncoeffs,), is_input=True, is_output=False),
                lp.GlobalArg("center, target",
                    shape=(dim,), is_input=True, is_output=False),
                lp.ValueArg("rscale", is_input=True),
                lp.ValueArg("itgt", is_input=True),
                lp.ValueArg("ntargets", is_input=True),
                lp.GlobalArg("targets",
                    shape=(dim, "ntargets"), is_input=True, is_output=False),
                *target_args,
                *temporary_variables,
                ...],
            name="e2p",
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
            fixed_parameters={
                "dim": dim,
                "nresults": len(kernels),
                "order": order,
                "max_deriv_order": max_deriv_order,
                "ncoeffs": ncoeffs,
            },
            )

    loopy_knl = lp.tag_inames(loopy_knl, "idim*:unr")

    for kernel in kernels:
        loopy_knl = kernel.prepare_loopy_kernel(loopy_knl)

    return loopy_knl, optimizations
