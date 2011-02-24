
from __future__ import division

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import sympy as sp
import sympy.printing.ccode
import numpy.linalg as la

# TODO:
# - Data layout, float4s bad
# - Make side-effect-free
# - Exclude self-interaction if source and target are same

# LATER:
# - Optimization for source = target (postpone)


DIRECT_KERNEL = """
    __kernel void sum_direct(
      __global ${pot_type} *potential_g,
      __global const ${coord_type}4 *target_g,
      __global const ${coord_type}4 *source_g,
      ulong nsource,
      ulong ntarget)
    {
      int itarget = get_global_id(0);
      if (itarget >= ntarget) return;

      ${coord_type}4 tgt = target_g[itarget];

      float p = 0;
      for(int isource=0; isource<nsource; isource++ )
      {
        %if exclude_self:
          if (isource == itarget)
            continue;
        %endif

        ${coord_type}4 src = source_g[isource];

        % for var_name, expr in vars_and_exprs:
          % if var_name == "result":
            p += ${expr};
          % else:
            ${pot_type} ${var_name} = ${expr};
          % endif
        % endfor
      }
      potential_g[itarget] = p;
    }
    """



def gen_target_source_subst_list(dimensions):
    result = {}
    for i in range(dimensions):
        result["s%d" % i] = "src.s%d" % i
        result["t%d" % i] = "tgt.s%d" % i

    return result




def gen_direct_sum_for_kernel(dimensions, expr, exclude_self=True):
    from exafmm.symbolic.codegen import generate_cl_statements_from_assignments
    vars_and_exprs = generate_cl_statements_from_assignments(
            [("result", expr)], 
            subst_map=gen_target_source_subst_list(dimensions))

    from mako.template import Template
    return Template(DIRECT_KERNEL).render(
            vars_and_exprs=vars_and_exprs,
            coord_type="float",
            pot_type="float",
            exclude_self="exclude_self")




def test_direct():
    # FIXME handle source strengths

    target = np.random.rand(5000, 4).astype(np.float32)
    source = np.random.rand(5000, 4).astype(np.float32)
    source[:,3] = 1

    dev = cl.get_platforms()[1].get_devices()[0]
    print dev.name
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)

    target_dev = cl_array.to_device(queue, target)
    source_dev = cl_array.to_device(queue, source)



    from exafmm.symbolic import make_coulomb_kernel_ts
    kernel_text = gen_direct_sum_for_kernel(3,
                    make_coulomb_kernel_ts(3)
                    #.diff(sp.Symbol("t0"))
                    )
    print kernel_text
    prg = cl.Program(ctx, kernel_text).build()

    sum_direct = prg.sum_direct
    sum_direct.set_scalar_arg_dtypes([None, None, None, np.uintp, np.uintp])

    potential_dev = cl_array.empty(queue, len(target), np.float32)
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

    assert la.norm(potential - potential_host)/la.norm(potential_host) < 1e-3




if __name__ == "__main__":
    #test_symbolic()
    test_direct()

# vim: foldmethod=marker
