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

import logging
import os
from functools import partial

import numpy as np
import pytest

from arraycontext import pytest_generate_tests_for_array_contexts

from sumpy.array_context import PytestPyOpenCLArrayContextFactory, _acf


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# Note: Do not import mpi4py.MPI object at the module level, because OpenMPI does not
# support recursive invocations.


def set_cache_dir(mpirank):
    """Make each rank use a different cache location to avoid conflict."""
    import platformdirs
    cache_dir = platformdirs.user_cache_dir("sumpy", "sumpy")

    # FIXME: should clean up this directory after running the tests
    os.environ["XDG_CACHE_HOME"] = os.path.join(cache_dir, str(mpirank))


# {{{ _test_against_single_rank

def _test_against_single_rank(
        dims, nsources, ntargets, dtype, communicate_mpoles_via_allreduce=False):
    from mpi4py import MPI

    # Get the current rank
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    set_cache_dir(mpi_rank)

    # Configure array context
    actx = _acf()

    def fmm_level_to_order(base_kernel, kernel_arg_set, tree, level):
        return max(level, 3)

    from boxtree.traversal import FMMTraversalBuilder
    traversal_builder = FMMTraversalBuilder(actx, well_sep_is_n_away=2)

    from sumpy.expansion import DefaultExpansionFactory
    from sumpy.kernel import LaplaceKernel
    kernel = LaplaceKernel(dims)
    expansion_factory = DefaultExpansionFactory()
    local_expansion_factory = expansion_factory.get_local_expansion_class(kernel)
    local_expansion_factory = partial(local_expansion_factory, kernel)
    multipole_expansion_factory = \
        expansion_factory.get_multipole_expansion_class(kernel)
    multipole_expansion_factory = partial(multipole_expansion_factory, kernel)

    from sumpy.fmm import SumpyTreeIndependentDataForWrangler
    tree_indep = SumpyTreeIndependentDataForWrangler(
        actx, multipole_expansion_factory, local_expansion_factory, [kernel])

    global_tree_dev = None
    sources_weights = actx.np.zeros(0, dtype=dtype)

    if mpi_rank == 0:
        # Generate random particles and source weights
        from boxtree.tools import make_normal_particle_array as p_normal
        sources = p_normal(actx, nsources, dims, dtype, seed=15)
        targets = p_normal(actx, ntargets, dims, dtype, seed=18)

        # FIXME: Use arraycontext instead of raw PyOpenCL arrays
        rng = np.random.default_rng(20)
        sources_weights = actx.from_numpy(rng.random(nsources, dtype=np.float64))
        target_radii = actx.from_numpy(0.05 * rng.random(ntargets, dtype=np.float64))

        # Build the tree and interaction lists
        from boxtree import TreeBuilder
        tb = TreeBuilder(actx)
        global_tree_dev, _ = tb(
            actx, sources, targets=targets, target_radii=target_radii,
            stick_out_factor=0.25, max_particles_in_box=30, debug=True)

        global_trav_dev, _ = traversal_builder(actx, global_tree_dev, debug=True)

        from sumpy.fmm import SumpyExpansionWrangler
        wrangler = SumpyExpansionWrangler(tree_indep, global_trav_dev, dtype,
                                          fmm_level_to_order)

        # Compute FMM with one MPI rank
        from boxtree.fmm import drive_fmm
        shmem_potential = drive_fmm(actx, wrangler, [sources_weights])

    # Compute FMM using the distributed implementation

    def wrangler_factory(local_traversal, global_traversal):
        from sumpy.distributed import DistributedSumpyExpansionWrangler
        return DistributedSumpyExpansionWrangler(
            actx, comm, tree_indep, local_traversal, global_traversal, dtype,
            fmm_level_to_order,
            communicate_mpoles_via_allreduce=communicate_mpoles_via_allreduce)

    from boxtree.distributed import DistributedFMMRunner
    distributed_fmm_info = DistributedFMMRunner(
        actx, global_tree_dev, traversal_builder, wrangler_factory, comm=comm)

    distributed_potential = distributed_fmm_info.drive_dfmm(actx, [sources_weights])

    if mpi_rank == 0:
        assert shmem_potential.shape == (1,)
        assert distributed_potential.shape == (1,)

        shmem_potential = actx.to_numpy(shmem_potential[0])
        distributed_potential = actx.to_numpy(distributed_potential[0])

        error = (np.linalg.norm(distributed_potential - shmem_potential, ord=np.inf)
                 / np.linalg.norm(shmem_potential, ord=np.inf))
        print(error)
        assert error < 1e-14


@pytest.mark.mpi
@pytest.mark.parametrize(
    "num_processes, dims, nsources, ntargets, communicate_mpoles_via_allreduce", [
        (4, 3, 10000, 10000, True),
        (4, 3, 10000, 10000, False)
    ]
)
def test_against_single_rank(
        num_processes, dims, nsources, ntargets, communicate_mpoles_via_allreduce):
    pytest.importorskip("mpi4py")

    from boxtree.tools import run_mpi
    run_mpi(__file__, num_processes, {
        "OMP_NUM_THREADS": 1,
        "_SUMPY_TEST_NAME": "against_single_rank",
        "_SUMPY_TEST_DIMS": dims,
        "_SUMPY_TEST_NSOURCES": nsources,
        "_SUMPY_TEST_NTARGETS": ntargets,
        "_SUMPY_TEST_MPOLES_ALLREDUCE": communicate_mpoles_via_allreduce
        })

# }}}


if __name__ == "__main__":
    if "_SUMPY_TEST_NAME" in os.environ:
        name = os.environ["_SUMPY_TEST_NAME"]
        if name == "against_single_rank":
            # Run "test_against_single_rank" test case
            dims = int(os.environ["_SUMPY_TEST_DIMS"])
            nsources = int(os.environ["_SUMPY_TEST_NSOURCES"])
            ntargets = int(os.environ["_SUMPY_TEST_NTARGETS"])

            communicate_mpoles_via_allreduce = (
                os.environ["_SUMPY_TEST_MPOLES_ALLREDUCE"] == "True")

            _test_against_single_rank(
                dims, nsources, ntargets, np.float64,
                communicate_mpoles_via_allreduce)
        else:
            raise ValueError(f"Invalid test name: {name!r}")
    else:
        # You can test individual routines by typing
        # $ python test_distributed.py
        # 'test_against_single_rank(4, 3, 10000, 10000, False)'
        import sys
        exec(sys.argv[1])
