from __future__ import division

import sympy as sp
import numpy as np
from mako.template import Template
import pyopencl as cl
import pyopencl.array as cl_array
from pytools import memoize_method

from exafmm.kernel_common import COMMON_PREAMBLE




# LATER:
# - Optimization for source == target (postpone)




P2P_KERNEL = Template(
    COMMON_PREAMBLE +
    """
    typedef ${geometry_type} geometry_t;
    typedef ${geometry_type}${dimensions} geometry_vec_t;
    typedef ${output_type} output_t;

    __kernel void ${name}(
      uint ntarget
      , uint nsource
      % for i in range(dimensions):
        , global const geometry_t *restrict t${i}_g
      % endfor
      % for i in range(dimensions):
        , global const geometry_t *restrict s${i}_g
      % endfor
      % for i in range(strength_count):
        , global const output_t *restrict strength${i}_g
      % endfor
      % for i in range(output_count):
        , global output_t *restrict output${i}_g
      % endfor
      )
    {
      int itarget = get_global_id(0);
      if (itarget >= ntarget) return;

      geometry_vec_t tgt;
      ${load_vector("tgt", "t", "itarget")}

      % for i in range(output_count):
        output_t output${i} = 0;
      %endfor

      for(int isource=0; isource<nsource; isource++ )
      {
        %if exclude_self:
          if (isource == itarget)
            continue;
        %endif

        geometry_vec_t src;
        ${load_vector("src", "s", "isource")}

        % for i in range(strength_count):
          output_t strength${i} = strength${i}_g[isource];
        % endfor

        % for var_name, expr in vars_and_exprs:
          % if var_name.startswith("output"):
            ${var_name} += ${expr};
          % else:
            output_t ${var_name} = ${expr};
          % endif
        % endfor
      }

      % for i in range(output_count):
        output${i}_g[itarget] = output${i};
      %endfor
    }
    """, strict_undefined=True, disable_unicode=True)




class P2PKernel(object):
    def __init__(self, ctx, dimensions, exprs, strength_usage=None, 
            exclude_self=True, options=[], name="p2p"):
        """
        :arg exprs: kernels which are to be evaluated for each source-target
          pair, as :mod:`sympy` expressions in terms of *t* and *s*.
        :arg strength_usage: A list of integers indicating which expression
          uses which source strength indicator. This implicitly specifies the
          number of strength arrays that need to be passed.
          Default: all kernels use the same strength.
        """
        self.context = ctx
        self.dimensions = dimensions
        self.exprs = exprs

        if strength_usage is None:
            strength_usage = [0] * len(exprs)

        if len(exprs) != len(strength_usage):
            raise ValueError("exprs and strength_usage must have the same length")

        self.strength_usage = strength_usage
        self.strength_count = max(strength_usage)+1

        self.exclude_self = exclude_self,
        self.options = options
        self.name = name

        self.max_wg_size = min(dev.max_work_group_size for dev in ctx.devices)

    @memoize_method
    def get_kernel(self, geometry_dtype, output_dtype):
        from exafmm.symbolic.codegen import (
                generate_cl_statements_from_assignments,
                gen_c_source_subst_map)
        vars_and_exprs = generate_cl_statements_from_assignments(
                [("output%d" % i, 
                    sp.Symbol("strength%d" % self.strength_usage[i])*expr)
                    for i, expr in enumerate(self.exprs)],
                subst_map=gen_c_source_subst_map(self.dimensions))

        from pyopencl.characterize import has_double_support
        from pyopencl.tools import dtype_to_ctype
        kernel_src = P2P_KERNEL.render(
                dimensions=self.dimensions,
                vars_and_exprs=vars_and_exprs,
                geometry_type=dtype_to_ctype(geometry_dtype),
                output_type=dtype_to_ctype(output_dtype),
                exclude_self=self.exclude_self,
                name=self.name,
                strength_count=self.strength_count,
                output_count=len(self.exprs),
                double_support=all(
                    has_double_support(dev) for dev in self.context.devices))

        prg = cl.Program(self.context, kernel_src).build(self.options)
        kernel = getattr(prg, self.name)
        kernel.set_scalar_arg_dtypes(
                [np.uint32, np.uint32]
                + [None]*( 2*self.dimensions + self.strength_count + len(self.exprs)))

        return kernel

    def __call__(self, targets, sources, src_strengths, queue=None, allocator=None):
        target_count, = targets[0].shape
        source_count, = sources[0].shape

        if targets[0].dtype != sources[0].dtype:
            raise TypeError("targets and sources must have same type")

        queue = queue or targets[0].queue or sources[0].queue
        allocator = allocator or targets[0].allocator

        if isinstance(src_strengths, cl_array.Array):
            src_strengths = [src_strengths]

        assert all(len(str_i) == source_count for str_i in src_strengths)

        output_dtype = src_strengths[0].dtype
        from pytools.obj_array import make_obj_array
        outputs = make_obj_array([
            cl_array.empty(queue, target_count, output_dtype)
            for expr in self.exprs])

        kernel = self.get_kernel(targets[0].dtype, output_dtype)

        wg_size = min(self.max_wg_size, 128) # FIXME: Tune?

        from pytools import div_ceil
        kernel(
                queue, (div_ceil(target_count, wg_size) * wg_size,), (wg_size,),
                target_count, source_count,
                *(
                    [tgt_i.data for tgt_i in targets]
                    + [src_i.data for src_i in sources]
                    + [str_i.data for str_i in src_strengths]
                    + [out_i.data for out_i in outputs]))

        return outputs
