from __future__ import division

import numpy as np
import numpy.linalg as la
import sympy as sp
import sys
import pytools.test




def have_cl():
    try:
        import pyopencl
        return True
    except:
        return False

if have_cl():
    import pyopencl.array as cl_array
    import pyopencl as cl
    from pyopencl.tools import pytest_generate_tests_for_pyopencl \
            as pytest_generate_tests




@pytools.test.mark_test.opencl
def test_p2p(ctx_getter):
    ctx = ctx_getter()

    print ctx.devices[0].name

    queue = cl.CommandQueue(ctx)

    dimensions = 3
    n = 5000

    from sumpy.symbolic import make_coulomb_kernel_ts
    coulomb_knl = make_coulomb_kernel_ts(dimensions)
    exprs = [coulomb_knl, coulomb_knl.diff(sp.Symbol("t0"))]

    from sumpy.p2p import P2PKernel
    knl = P2PKernel(ctx, dimensions, exprs, exclude_self=False)

    targets = np.random.rand(dimensions, n).astype(np.float32)
    sources = np.random.rand(dimensions, n).astype(np.float32)

    from sumpy.tools import vector_to_device
    targets_dev = vector_to_device(queue, targets)
    sources_dev = vector_to_device(queue, sources)
    strengths_dev = cl_array.empty(queue, n, dtype=np.float32)
    strengths_dev.fill(1)

    potential_dev, x_derivative = knl(targets_dev, sources_dev, strengths_dev)

    potential = potential_dev.get()
    potential_host = np.empty_like(potential)
    strengths = strengths_dev.get()

    targets = targets.T
    sources = sources.T
    for itarg in xrange(n):
        potential_host[itarg] = np.sum(
                strengths
                /
                np.sum((targets[itarg] - sources)**2, axis=-1)**0.5)

    rel_err = la.norm(potential - potential_host)/la.norm(potential_host)
    assert rel_err < 1e-3




@pytools.test.mark_test.opencl
def test_m2p(ctx_getter):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    dimensions = 3
    n = 5000

    from sumpy.symbolic import make_coulomb_kernel_in
    from sumpy.expansion import TaylorExpansion
    texp = TaylorExpansion(
            make_coulomb_kernel_in("b", dimensions),
            order=2, dimensions=dimensions)

    from sumpy.m2p import M2PKernel
    knl = M2PKernel(ctx, texp,
            output_maps=[
                lambda expr: expr,
                lambda expr: sp.diff(expr, sp.Symbol("t0"))])

    targets = np.random.rand(dimensions, n).astype(np.float32)

    from sumpy.tools import vector_to_device
    targets_dev = vector_to_device(queue, targets)

    knl(targets_dev, None, None, targets_dev[0], None, None)





# You can test individual routines by typing
# $ python test_kernels.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pyopencl as cl

    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
