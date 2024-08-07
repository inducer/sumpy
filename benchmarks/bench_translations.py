from __future__ import annotations

import logging

import numpy as np

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

from sumpy.expansion.local import (
    H2DLocalExpansion,
    LinearPDEConformingVolumeTaylorLocalExpansion,
    VolumeTaylorLocalExpansion,
)
from sumpy.expansion.multipole import (
    H2DMultipoleExpansion,
    LinearPDEConformingVolumeTaylorMultipoleExpansion,
    VolumeTaylorMultipoleExpansion,
)
from sumpy.kernel import HelmholtzKernel, LaplaceKernel


logger = logging.getLogger(__name__)

import pymbolic.mapper.flop_counter

import sumpy.symbolic as sym
from sumpy.assignment_collection import SymbolicAssignmentCollection
from sumpy.codegen import to_loopy_insns


class Param:
    def __init__(self, dim, order):
        self.dim = dim
        self.order = order

    def __repr__(self):
        return f"{self.dim}D_order_{self.order}"


class TranslationBenchmarkSuite:

    params = (
        Param(2, 10),
        Param(2, 15),
        Param(2, 20),
        Param(3, 5),
        Param(3, 10),
    )

    param_names = ("order",)

    def setup(self, param):
        logging.basicConfig(level=logging.INFO)
        np.random.seed(17)  # noqa: NPY002
        if self.__class__ == TranslationBenchmarkSuite:
            raise NotImplementedError
        mpole_expn_class = self.mpole_expn_class
        if param.order == 3 and H2DMultipoleExpansion == mpole_expn_class:
            raise NotImplementedError

    def track_m2l_op_count(self, param):
        knl = self.knl(param.dim)
        m_expn = self.mpole_expn_class(knl, order=param.order)
        l_expn = self.local_expn_class(knl, order=param.order)

        src_coeff_exprs = [
            sym.Symbol(f"src_coeff{i}")
            for i in range(len(m_expn))]
        dvec = sym.make_sym_vector("d", knl.dim)
        src_rscale = sym.Symbol("src_rscale")
        tgt_rscale = sym.Symbol("tgt_rscale")
        sac = SymbolicAssignmentCollection()
        try:
            result = l_expn.translate_from(m_expn, src_coeff_exprs, src_rscale,
                                       dvec, tgt_rscale, sac)
        except TypeError:
            # Support older interface to make it possible to compare
            # in CI run
            result = l_expn.translate_from(m_expn, src_coeff_exprs, src_rscale,
                                       dvec, tgt_rscale)
        for i, expr in enumerate(result):
            sac.assign_unique(f"coeff{i}", expr)
        sac.run_global_cse()
        insns = to_loopy_insns(sac.assignments.items())
        counter = pymbolic.mapper.flop_counter.CSEAwareFlopCounter()

        return sum(counter.rec(insn.expression)+1 for insn in insns)

    track_m2l_op_count.unit = "ops"
    track_m2l_op_count.timeout = 300.0


class LaplaceVolumeTaylorTranslation(TranslationBenchmarkSuite):
    knl = LaplaceKernel
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    params = (
        Param(2, 10),
        Param(3, 5),
    )


class LaplaceConformingVolumeTaylorTranslation(TranslationBenchmarkSuite):
    knl = LaplaceKernel
    local_expn_class = LinearPDEConformingVolumeTaylorLocalExpansion
    mpole_expn_class = LinearPDEConformingVolumeTaylorMultipoleExpansion


class HelmholtzVolumeTaylorTranslation(TranslationBenchmarkSuite):
    knl = HelmholtzKernel
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    params = (
        Param(2, 10),
        Param(3, 5),
    )


class HelmholtzConformingVolumeTaylorTranslation(TranslationBenchmarkSuite):
    knl = HelmholtzKernel
    local_expn_class = LinearPDEConformingVolumeTaylorLocalExpansion
    mpole_expn_class = LinearPDEConformingVolumeTaylorMultipoleExpansion


class Helmholtz2DTranslation(TranslationBenchmarkSuite):
    knl = HelmholtzKernel
    local_expn_class = H2DLocalExpansion
    mpole_expn_class = H2DMultipoleExpansion
    params = (
        Param(2, 10),
        Param(2, 15),
        Param(2, 20),
    )
