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

import pymbolic
import loopy as lp
import numpy as np
import sumpy.symbolic as sym
from sumpy.symbolic import SympyToPymbolicMapper
from sumpy.assignment_collection import SymbolicAssignmentCollection
from sumpy.tools import gather_loopy_arguments

import logging
logger = logging.getLogger(__name__)


def e2p_loopy_knl_expr(expansion, kernels):
    """
    This is a helper function to create a loopy kernel for multipole/local
    evaluation. This function uses symbolic expressions given by the expansion class,
    converts them to pymbolic expressions and generates a loopy
    kernel. Note that the loopy kernel returned has lots of expressions in it and
    takes a long time. Therefore, this function should be used only as a fallback
    when there is no "loop-y" kernel to evaluate the expansion
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

    sympy_conv = SympyToPymbolicMapper()
    kernel_scaling = sym.Symbol("kernel_scaling")
    insns.append(
        lp.Assignment(
            id=None,
            assignee="kernel_scaling",
            expression=sympy_conv(expansion.kernel.get_global_scaling_const()),
            temp_var_type=lp.Optional(None),
        )
    )

    coeff_exprs = sym.make_sym_vector("coeffs", ncoeffs)
    coeff_names = [
        sac.add_assignment(f"result{i}",
            expansion.evaluate(knl, coeff_exprs, bvec, rscale, sac=sac)
                    * kernel_scaling)
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

    return loopy_knl
