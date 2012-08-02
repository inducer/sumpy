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
    particles = make_obj_array([
        rng.normal(queue, nparticles, dtype=dtype)
        for i in range(dims)])

    do_plot = 0

    if do_plot:
        pt.plot(particles[0].get(), particles[1].get(), "x")

    from sumpy.tree import TreeBuilder
    tb = TreeBuilder(ctx)

    queue.finish()
    print "building..."
    tree = tb(queue, particles, max_particles_in_box=30)
    print "%d boxes, testing..." % tree.nboxes

    starts = tree.box_starts.get()
    pcounts = tree.box_particle_counts.get()
    sorted_particles = np.array([pi.get() for pi in tree.particles])
    centers = tree.box_centers.get()
    sizes = tree.box_sizes.get()


    for ibox in xrange(tree.nboxes):
        el = extent_low = centers[:, ibox] - sizes[ibox]*0.5
        eh = extent_high = extent_low + sizes[ibox]

        box_particle_nrs = np.arange(starts[ibox], starts[ibox]+pcounts[ibox],
                dtype=np.intp)

        if do_plot:
            pt.plot([el[0], eh[0], eh[0], el[0], el[0]],
                    [el[1], el[1], eh[1], eh[1], el[1]], "-")

        assert (sorted_particles[:,box_particle_nrs] < extent_high[:, np.newaxis]).all()
        assert (extent_low[:, np.newaxis] <= sorted_particles[:,box_particle_nrs]).all()

    print "done"

    if do_plot:
        pt.gca().set_aspect("equal", "datalim")
        pt.show()









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
