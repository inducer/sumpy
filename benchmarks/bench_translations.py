import numpy as np

import pytest
import pyopencl as cl
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from sumpy.expansion.multipole import (
        VolumeTaylorMultipoleExpansion, H2DMultipoleExpansion,
        VolumeTaylorMultipoleExpansionBase,
        LaplaceConformingVolumeTaylorMultipoleExpansion,
        HelmholtzConformingVolumeTaylorMultipoleExpansion)
from sumpy.expansion.local import (
        VolumeTaylorLocalExpansion, H2DLocalExpansion,
        LaplaceConformingVolumeTaylorLocalExpansion,
        HelmholtzConformingVolumeTaylorLocalExpansion)

from sumpy.kernel import (LaplaceKernel, HelmholtzKernel, AxisTargetDerivative,
        DirectionalSourceDerivative)

import logging
logger = logging.getLogger(__name__)

import sympy
import six
import pymbolic.mapper.flop_counter

import sumpy.symbolic as sym
from sumpy.assignment_collection import SymbolicAssignmentCollection
from sumpy.codegen import to_loopy_insns

class Param:
    def __init__(self, dim, order):
        self.dim = dim
        self.order = order

    def __repr__(self):
        return "{}D_order_{}".format(self.dim, self.order)


class TranslationBenchmarkSuite:

    params = [
        Param(2, 10),
        Param(2, 15),
        Param(2, 20),
        Param(3, 5),
        Param(3, 10),
    ]

    param_names = ['order']

    def setup(self, param):
        logging.basicConfig(level=logging.INFO)
        np.random.seed(17)
        if self.__class__ == TranslationBenchmarkSuite:
            raise NotImplementedError
        mpole_expn_class = self.mpole_expn_class
        if param.order == 3 and H2DMultipoleExpansion == mpole_expn_class:
            raise NotImplementedError

    def track_m2l_op_count(self, param):
        knl = self.knl(param.dim)
        m_expn = self.mpole_expn_class(knl, order=param.order)
        l_expn = self.local_expn_class(knl, order=param.order)

        src_coeff_exprs = [sym.Symbol("src_coeff%d" % i)
                for i in range(len(m_expn))]
        dvec = sym.make_sym_vector("d", knl.dim)
        src_rscale = sym.Symbol("src_rscale")
        tgt_rscale = sym.Symbol("tgt_rscale")
        result = l_expn.translate_from(m_expn, src_coeff_exprs, src_rscale,
                                       dvec, tgt_rscale)
        sac = SymbolicAssignmentCollection()
        for i, expr in enumerate(result):
            sac.assign_unique("coeff%d" % i, expr)
        sac.run_global_cse()
        insns, _ = to_loopy_insns(six.iteritems(sac.assignments))
        counter = pymbolic.mapper.flop_counter.CSEAwareFlopCounter()

        return sum([counter.rec(insn.expression)+1 for insn in insns])

    track_m2l_op_count.unit = "ops"
    track_m2l_op_count.timeout = 200.0


class LaplaceVolumeTaylorTranslation(TranslationBenchmarkSuite):
    knl = LaplaceKernel
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    params = [
        Param(2, 10),
        Param(3, 5),
    ]


class LaplaceConformingVolumeTaylorTranslation(TranslationBenchmarkSuite):
    knl = LaplaceKernel
    local_expn_class = LaplaceConformingVolumeTaylorLocalExpansion
    mpole_expn_class = LaplaceConformingVolumeTaylorMultipoleExpansion


class HelmholtzVolumeTaylorTranslation(TranslationBenchmarkSuite):
    knl = HelmholtzKernel
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    params = [
        Param(2, 10),
        Param(3, 5),
    ]


class HelmholtzConformingVolumeTaylorTranslation(TranslationBenchmarkSuite):
    knl = HelmholtzKernel
    local_expn_class = HelmholtzConformingVolumeTaylorLocalExpansion
    mpole_expn_class = HelmholtzConformingVolumeTaylorMultipoleExpansion


class Helmholtz2DTranslation(TranslationBenchmarkSuite):
    knl = HelmholtzKernel
    local_expn_class = H2DLocalExpansion
    mpole_expn_class = H2DMultipoleExpansion
    params = [
        Param(2, 10),
        Param(2, 15),
        Param(2, 20),
    ]


