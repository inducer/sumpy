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

from boxtree.distributed.calculation import DistributedExpansionWrangler

import pyopencl.array as cl_array
import pytools.obj_array as obj_array

from sumpy.fmm import SumpyExpansionWrangler


class DistributedSumpyExpansionWrangler(
        DistributedExpansionWrangler, SumpyExpansionWrangler):
    def __init__(
            self, context, comm, tree_indep, local_traversal, global_traversal,
            dtype, fmm_level_to_order, communicate_mpoles_via_allreduce=False,
            **kwarg):
        DistributedExpansionWrangler.__init__(
            self, context, comm, global_traversal, True,
            communicate_mpoles_via_allreduce=communicate_mpoles_via_allreduce)
        SumpyExpansionWrangler.__init__(
            self, tree_indep, local_traversal, dtype, fmm_level_to_order, **kwarg)

    def distribute_source_weights(self, src_weight_vecs, src_idx_all_ranks):
        src_weight_vecs_host = [src_weight.get() for src_weight in src_weight_vecs]

        local_src_weight_vecs_host = super().distribute_source_weights(
            src_weight_vecs_host, src_idx_all_ranks)

        local_src_weight_vecs_device = [
            cl_array.to_device(src_weight.queue, local_src_weight)
            for local_src_weight, src_weight in
            zip(local_src_weight_vecs_host, src_weight_vecs, strict=True)]

        return local_src_weight_vecs_device

    def gather_potential_results(self, potentials, tgt_idx_all_ranks):
        mpi_rank = self.comm.Get_rank()

        potentials_host_vec = [potentials_dev.get() for potentials_dev in potentials]

        gathered_potentials_host_vec = []
        for potentials_host in potentials_host_vec:
            gathered_potentials_host_vec.append(
                super().gather_potential_results(potentials_host, tgt_idx_all_ranks))

        if mpi_rank == 0:
            return obj_array.new_1d([
                cl_array.to_device(potentials_dev.queue, gathered_potentials_host)
                for gathered_potentials_host, potentials_dev in
                zip(gathered_potentials_host_vec, potentials, strict=True)])
        else:
            return None

    def reorder_sources(self, source_array):
        if self.comm.Get_rank() == 0:
            return source_array.with_queue(source_array.queue)[
                self.global_traversal.tree.user_source_ids]
        else:
            return source_array

    def reorder_potentials(self, potentials):
        if self.comm.Get_rank() == 0:
            import numpy as np

            assert (
                    isinstance(potentials, np.ndarray)
                    and potentials.dtype.char == "O")

            def reorder(x):
                return x[self.global_traversal.tree.sorted_target_ids]

            return obj_array.vectorize(reorder, potentials)
        else:
            return None

    def communicate_mpoles(self, mpole_exps, return_stats=False):
        mpole_exps_host = mpole_exps.get()
        stats = super().communicate_mpoles(mpole_exps_host, return_stats)
        mpole_exps[:] = mpole_exps_host
        return stats
