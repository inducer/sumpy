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
    def __init__(self, knl, local_expn_class, mpole_expn_class):
        self.knl = knl
        self.local_expn_class = local_expn_class
        self.mpole_expn_class = mpole_expn_class

    def __repr__(self):
        return "{}_{}_{}".format(self.knl, self.local_expn_class.__name__, self.mpole_expn_class.__name__)


class TranslationSuite:

    params = [
        Param(LaplaceKernel(2), VolumeTaylorLocalExpansion, VolumeTaylorMultipoleExpansion),
        Param(LaplaceKernel(2), LaplaceConformingVolumeTaylorLocalExpansion,
         LaplaceConformingVolumeTaylorMultipoleExpansion),
        Param(HelmholtzKernel(2), VolumeTaylorLocalExpansion, VolumeTaylorMultipoleExpansion),
        Param(HelmholtzKernel(2), HelmholtzConformingVolumeTaylorLocalExpansion,
         HelmholtzConformingVolumeTaylorMultipoleExpansion),
        Param(HelmholtzKernel(2), H2DLocalExpansion, H2DMultipoleExpansion)
    ]
    param_names = ['translation']

    def setup(self, param):
        logging.basicConfig(level=logging.INFO)
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        np.random.seed(17)

    def track_m2l_op_count(self, param):
        knl = param.knl
        m_expn = param.mpole_expn_class(knl, order=3)
        l_expn = param.local_expn_class(knl, order=3)

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
        insns = to_loopy_insns(six.iteritems(sac.assignments))
        counter = pymbolic.mapper.flop_counter.FlopCounter()

        return sum([counter.rec(insn.expression)+1 for insn in insns])

    track_m2l_op_count.unit = "ops"
