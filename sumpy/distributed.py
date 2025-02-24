from __future__ import annotations


__copyright__ = "Copyright (C) 2022 Hao Gao"

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

from boxtree.distributed.calculation import DistributedExpansionWranglerMixin

from sumpy.array_context import PyOpenCLArrayContext
from sumpy.fmm import SumpyExpansionWrangler


class DistributedSumpyExpansionWrangler(
        DistributedExpansionWranglerMixin, SumpyExpansionWrangler):
    def __init__(
            self, actx: PyOpenCLArrayContext,
            comm, tree_indep, local_traversal, global_traversal,
            dtype, fmm_level_to_order, communicate_mpoles_via_allreduce=False,
            **kwargs):
        SumpyExpansionWrangler.__init__(
            self, tree_indep, local_traversal, dtype, fmm_level_to_order,
            **kwargs)

        self.comm = comm
        self.traversal_in_device_memory = True
        self.global_traversal = global_traversal
        self.communicate_mpoles_via_allreduce = communicate_mpoles_via_allreduce

    def distribute_source_weights(self,
            actx: PyOpenCLArrayContext, src_weight_vecs, src_idx_all_ranks):
        src_weight_vecs_host = [
            actx.to_numpy(src_weight) for src_weight in src_weight_vecs
            ]

        local_src_weight_vecs_host = super().distribute_source_weights(
            actx, src_weight_vecs_host, src_idx_all_ranks)

        local_src_weight_vecs_device = [
            actx.from_numpy(local_src_weight)
            for local_src_weight in local_src_weight_vecs_host]

        return local_src_weight_vecs_device

    def gather_potential_results(self,
            actx: PyOpenCLArrayContext, potentials, tgt_idx_all_ranks):
        potentials_host_vec = [
            actx.to_numpy(potentials_dev) for potentials_dev in potentials
            ]

        gathered_potentials_host_vec = []
        for potentials_host in potentials_host_vec:
            gathered_potentials_host_vec.append(
                super().gather_potential_results(
                    actx, potentials_host, tgt_idx_all_ranks))

        if self.is_mpi_root:
            from pytools.obj_array import make_obj_array
            return make_obj_array([
                actx.from_numpy(gathered_potentials_host)
                for gathered_potentials_host in gathered_potentials_host_vec
                ])
        else:
            return None

    def reorder_sources(self, source_array):
        if self.is_mpi_root:
            return source_array[self.global_traversal.tree.user_source_ids]
        else:
            return source_array

    def reorder_potentials(self, potentials):
        if self.is_mpi_root:
            import numpy as np

            from pytools.obj_array import obj_array_vectorize
            assert (
                    isinstance(potentials, np.ndarray)
                    and potentials.dtype.char == "O")

            def reorder(x):
                return x[self.global_traversal.tree.sorted_target_ids]

            return obj_array_vectorize(reorder, potentials)
        else:
            return None

    def communicate_mpoles(self,
            actx: PyOpenCLArrayContext, mpole_exps, return_stats=False):
        mpole_exps_host = actx.to_numpy(mpole_exps)
        stats = super().communicate_mpoles(actx, mpole_exps_host, return_stats)
        mpole_exps[:] = mpole_exps_host
        return stats
