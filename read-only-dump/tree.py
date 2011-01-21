from __future__ import division

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import numpy.linalg as la

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


def test_tree():
    import logging
    from math import log
    logging.basicConfig(filename = 'fmm.log', level = logging.DEBUG)

    logging.debug("CTX")
    dev = cl.get_platforms()[0].get_devices()[1]
    logging.debug(dev.name)
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    logging.debug("CTX END")

    logging.debug('This message should go to the log file')
    order           = 3
    numCoefficients = order*(order+1)*(order+2)/6
    
    separationMax  = 10
    base           = 1.55
    targetOffsets  = base**np.linspace(log(1, base), log(separationMax, base), 20)
    res            = []
    np.random.seed(1)
    # Convergence study
    for target_offset in targetOffsets:
        # (x,y,z,phi) for each target
        xt = yt = zt = 0.5 - target_offset
        target = np.random.rand(50, 4).astype(np.float32)
        target[:,0] += xt - 0.5
        target[:,1] += yt - 0.5
        target[:,2] += zt - 0.5
        # (x,y,z,q) for each source
        source = np.random.rand(50, 4).astype(np.float32)
        # M_i: coefficients of multipole expansion
        multip = np.zeros((numCoefficients,1))

        # Expand around source box center
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

        target_dev = cl_array.to_device(queue, target)
        source_dev = cl_array.to_device(queue, source)

        prg = cl.Program(ctx,DIRECT_KERNEL).build()

        sum_direct = prg.sum_direct
        sum_direct.set_scalar_arg_dtypes([None, None, None, np.uintp, np.uintp])

        potential_dev = cl_array.empty(queue, len(target), np.float32)
        grp_size = 1
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
            potential[i] = p
        for itarg in xrange(len(target)):
            potential_host[itarg] = np.sum(
                source[:,3]
                /
                np.sum((target[itarg,:3] - source[:,:3])**2, axis=-1)**0.5)

        logging.debug(potential[:10])
        logging.debug(potential_host[:10])
        residual = la.norm(potential - potential_host)/la.norm(potential_host)
        logging.debug('Potential Residual: %g' % residual)
        res.append(residual)
    res = np.array(res)
    logging.debug(res)
    dist = np.sqrt(3*targetOffsets**2)
    intercept, slope = np.polyfit(np.log(dist), np.log(res), 1)
    if abs(slope + order+1) > 1.0e-1:
        import sys
        sys.exit('Order of approximation should be %d' % order+1)
    return

if __name__ == "__main__":
    test_tree()

# vim: foldmethod=marker
