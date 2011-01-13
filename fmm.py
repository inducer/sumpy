from __future__ import division

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import numpy.linalg as la

# TODO:
# - Data layout, float4s bad
# - Make side-effect-free
# - Exclude self-interaction if source and target are same

# LATER:
# - Optimization for source = target (postpone)


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




def test_direct():
    target = np.random.rand(5000, 4).astype(np.float32)
    source = np.random.rand(5000, 4).astype(np.float32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    target_dev = cl_array.to_device(ctx, queue, target)
    source_dev = cl_array.to_device(ctx, queue, source)

    prg = cl.Program(ctx, DIRECT_KERNEL).build()
    sum_direct = prg.sum_direct
    sum_direct.set_scalar_arg_dtypes([None, None, None, np.uintp, np.uintp])

    potential_dev = cl_array.empty(ctx, len(target), np.float32, queue=queue)
    grp_size = 128
    sum_direct(queue, ((len(target) + grp_size) // grp_size * grp_size,), (grp_size,), 
	potential_dev.data, target_dev.data, source_dev.data, len(source), len(target))

    potential = potential_dev.get()
    potential_host = np.empty_like(potential)

    for itarg in xrange(len(target)):
	potential_host[itarg] = np.sum(
		source[:,3]
		/
		np.sum((target[itarg,:3] - source[:,:3])**2, axis=-1)**0.5)

    #print potential[:100]
    #print potential_host[:100]
    assert la.norm(potential - potential_host)/la.norm(potential_host) < 1e-6


    	



if __name__ == "__main__":
    test_direct()
