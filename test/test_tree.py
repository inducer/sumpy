from __future__ import division

import numpy as np
import sys
import pytools.test

import matplotlib.pyplot as pt





@pytools.test.mark_test.opencl
def test_tree(ctx_getter):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    dims = 2
    nparticles = 1000000
    dtype = np.float64

    from pyopencl.clrandom import RanluxGenerator
    rng = RanluxGenerator(queue, seed=15)

    from pytools.obj_array import make_obj_array
    points = make_obj_array([
        rng.normal(queue, nparticles, dtype=dtype)
        for i in range(dims)])

    #pt.plot(points[0].get(), points[1].get(), "x")
    #pt.show()

    from sumpy.tree import TreeBuilder
    tb = TreeBuilder(ctx)
    tb(queue, points, max_particles_in_box=30)







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

# vim: fdm=marker
