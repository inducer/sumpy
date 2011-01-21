from __future__ import division

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import numpy.linalg as la
import sympy as sp

DIRECT_KERNEL = """
    __kernel void sum_direct(
      __global float *potential_g,
      __global const float4 *target_g,
      __global const float4 *source_g,
      ulong nsource,
      ulong ntarget)
    {
      int itarget = get_global_id(0);
      if (itarget >= ntarget) return;

      float p=0;
      for(int isource=0; isource<nsource; isource++ )
      {
        float4 dist = target_g[itarget] - source_g[isource];
        float4 dist_sq = dist*dist;
        p += source_g[isource].w * rsqrt(dist_sq.x + dist_sq.y + dist_sq.z);
      }
      potential_g[itarget] = p;
    }
    """




def generate_derivatives(dimensions, max_order):
    from sumpy.fmm import make_sym_vector
    x = make_sym_vector("x", dimensions)
    func = 1/sp.sqrt((x.T*x)[0,0])

    yield func

    derivative_cache = {
            dimensions*(0,): func
            }
    for order in range(max_order+1):
        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)
        for idx in gnitstam(order, dimensions):







def test_direct():
    target = -np.random.rand(50, 4).astype(np.float32)
    source = np.random.rand(50, 4).astype(np.float32)
    multip = np.zeros((10,1))

    xc = yc = zc = 0.5
    for j in range(len(source)):
        dx = xc-source[j,0]
        dy = yc-source[j,1]
        dz = zc-source[j,2]
        multip[0] += source[j,3]
        multip[1] += source[j,3] * dx
        multip[2] += source[j,3] * dy
        multip[3] += source[j,3] * dz
        multip[4] += source[j,3] * dx * dx / 2
        multip[5] += source[j,3] * dy * dy / 2
        multip[6] += source[j,3] * dz * dz / 2
        multip[7] += source[j,3] * dx * dy / 2
        multip[8] += source[j,3] * dy * dz / 2
        multip[9] += source[j,3] * dz * dx / 2
        # this one is \vec(x)^n / n!

    print "CTX"
    dev = cl.get_platforms()[0].get_devices()[1]
    print dev.name
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    print "CTX END"

    target_dev = cl_array.to_device(ctx, queue, target)
    source_dev = cl_array.to_device(ctx, queue, source)


    prg = cl.Program(ctx,DIRECT_KERNEL).build()

    sum_direct = prg.sum_direct
    sum_direct.set_scalar_arg_dtypes([None, None, None, np.uintp, np.uintp])

    potential_dev = cl_array.empty(ctx, len(target), np.float32, queue=queue)
    grp_size = 128
    sum_direct(queue, ((len(target) + grp_size) // grp_size * grp_size,), (grp_size,),
        potential_dev.data, target_dev.data, source_dev.data, len(source), len(target))

    potential = potential_dev.get()
    potential_host = np.empty_like(potential)

    for i in range(len(target)):
        p = 0
        X = target[i,0] - xc
        Y = target[i,1] - yc
        Z = target[i,2] - zc
        R = (X * X + Y * Y + Z * Z)**0.5
        R3 = R * R * R
        R5 = R3 * R * R
        p += multip[0] / R
        p += multip[1] * (-X / R3)
        p += multip[2] * (-Y / R3)
        p += multip[3] * (-Z / R3)
        p += multip[4] * (3 * X * X / R5 - 1 / R3)
        p += multip[5] * (3 * Y * Y / R5 - 1 / R3)
        p += multip[6] * (3 * Z * Z / R5 - 1 / R3)
        p += multip[7] * (3 * X * Y / R5)
        p += multip[8] * (3 * Y * Z / R5)
        p += multip[9] * (3 * Z * X / R5)
        # this one is grad^n 1/R
        # ok -- i'll go play with sympy on screen 1
        potential[i] = p
    for itarg in xrange(len(target)):
        potential_host[itarg] = np.sum(
                source[:,3]
                /
                np.sum((target[itarg,:3] - source[:,:3])**2, axis=-1)**0.5)

    print potential[:10]
    print potential_host[:10]
    print la.norm(potential - potential_host)/la.norm(potential_host)






if __name__ == "__main__":
    test_direct()

# vim: foldmethod=marker
