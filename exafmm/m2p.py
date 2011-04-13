from __future__ import division

import numpy as np
from mako.template import Template
from pytools import memoize_method

import pyopencl as cl
import pyopencl.array as cl_array

from exafmm.kernel_common import COMMON_PREAMBLE



M2P_KERNEL = Template(
COMMON_PREAMBLE +
"""
typedef uint offset_t;
typedef uint mpole_offset_t;
typedef ${coefficient_type} coeff_t;
typedef ${geometry_type} geometry_t;
typedef ${geometry_type}${dimensions} geometry_vec_t;
typedef ${output_type} output_t;




#define TGT_CELL get_group_id(0)
<% wg_size = coef_cnt_padded*par_cell_cnt %>

__kernel
__attribute__((reqd_work_group_size(${coef_cnt_padded}, ${par_cell_cnt}, 1)))
void m2p(
% for i in range(dimensions):
  global const geometry_t *t${i}_g,
% endfor
% for i in range(output_count):
  global output_t *output${i}_g,
% endfor
  global const offset_t *m2p_ilist_starts_g,
  global const mpole_offset_t *m2p_ilist_mpole_offsets_g,
  global const coeff_t *mpole_coeff_g,
  global const offset_t *cell_idx_to_particle_offset_g,
  global const uint *cell_idx_to_particle_cnt_g
  )

{
  uint lid = get_local_id(0) + ${coef_cnt_padded} * get_local_id(1);

  offset_t ilist_start = m2p_ilist_starts_g[TGT_CELL];
  offset_t ilist_end = m2p_ilist_starts_g[TGT_CELL+1];

  offset_t tgt_cell_particle_offset = cell_idx_to_particle_offset_g[TGT_CELL];
  uint tgt_cell_particle_count = cell_idx_to_particle_cnt_g[TGT_CELL];

  __local coeff_t mpole_coeff_l[${par_cell_cnt} * ${coef_cnt_padded}];

  // index into this cell's list of particles
  uint plist_idx = lid;

  // loop over particle batches
  while (plist_idx < tgt_cell_particle_count)
  {
    geometry_vec_t tgt;
    ${load_vector("tgt", "t", "tgt_cell_particle_offset + plist_idx")}

    % for i in range(output_count):
      output_t output${i} = 0;
    % endfor

    // index into overall M2P interaction list
    offset_t ilist_idx = ilist_start + get_local_id(1);

    // loop over source cell batches
    while (ilist_idx < ilist_end)
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      mpole_offset_t mpole_offset = m2p_ilist_mpole_offsets_g[ilist_idx];
      mpole_coeff_l[lid] = mpole_coeff_g[mpole_offset + get_local_id(0)];
      barrier(CLK_LOCAL_MEM_FENCE);

      for (uint src_cell = 0; src_cell < ${par_cell_cnt}; ++src_cell)
      {
        uint loc_mpole_base = src_cell*${coef_cnt_padded};
        geometry_vec_t ctr;
        % for i in range(dimensions):
          ctr.s${i} = mpole_coeff_l[loc_mpole_base+${i}];
        % endfor
        loc_mpole_base += ${dimensions};

        % for var, expr in vars_and_exprs:
          % if var.startswith("output"):
            ${var} = ${expr};
          % else:
            output_t ${var} = ${expr};
          % endif
        % endfor
      }

      ilist_idx += ${par_cell_cnt};
    }

    % for i in range(output_count):
      output${i}_g[tgt_cell_particle_offset + plist_idx] = output${i};
    % endfor

    plist_idx += ${wg_size};
  }
}
""", strict_undefined=True, disable_unicode=True)




class M2PKernel(object):
    def __init__(self, ctx, expansion, output_maps=[lambda x: x],
            options=[], name="m2p"):
        """
        :output_maps: A list of functions which will be applied to basis functions
          in the expansion, each generating a different output array.
        """
        self.context = ctx
        self.expansion = expansion
        self.output_maps = output_maps

        self.options = options
        self.name = name

    @memoize_method
    def get_kernel(self, geometry_dtype, coeff_dtype, output_dtype):
        coeff_dtype = np.dtype(coeff_dtype)

        dimensions = self.expansion.dimensions

        from exafmm.symbolic.codegen import generate_cl_statements_from_assignments
        from exafmm.symbolic import vector_subs, make_sym_vector

        old_var = make_sym_vector("b", dimensions)
        new_var = (make_sym_vector("t", dimensions)
                - make_sym_vector("c", dimensions))

        tc_basis = [
                vector_subs(basis_fun, old_var, new_var)
                for basis_fun in self.expansion.basis]

        outputs = [
                ("output%d" % output_idx,
                    output_map(sum(tc_basis_fun for tc_basis_fun in tc_basis)))

                for output_idx, output_map in enumerate(self.output_maps)
                ]

        from exafmm.symbolic.codegen import gen_c_source_subst_map
        subst_map = gen_c_source_subst_map(dimensions)

        vars_and_exprs = generate_cl_statements_from_assignments(
                outputs, subst_map=subst_map)

        from pyopencl.characterize import has_double_support
        from pyopencl.tools import dtype_to_ctype
        kernel_src = M2P_KERNEL.render(
                dimensions=dimensions,
                vars_and_exprs=vars_and_exprs,
                type_declarations="",
                par_cell_cnt=10,
                coef_cnt_padded=self.expansion.padded_coefficient_count(
                    coeff_dtype),
                output_count=len(self.output_maps),
                double_support=all(
                    has_double_support(dev) for dev in self.context.devices),

                coefficient_type=dtype_to_ctype(coeff_dtype),
                geometry_type=dtype_to_ctype(geometry_dtype),
                output_type=dtype_to_ctype(output_dtype),
                )
        print kernel_src

        prg = cl.Program(self.context, kernel_src).build(self.options)
        kernel = getattr(prg, self.name)
        kernel.set_scalar_arg_dtypes(
                [None]*(dimensions + len(self.output_maps) + 5))

        return kernel

    def __call__(self, targets, m2p_ilist_starts, m2p_ilist_mpole_offsets,
            mpole_coeff,
            cell_idx_to_particle_offset, cell_idx_to_particle_cnt,
            queue=None, allocator=None, output_dtype=None):
        target_count, = targets[0].shape

        queue = queue or targets[0].queue
        allocator = allocator or targets[0].allocator

        geometry_dtype = targets[0].dtype
        coeff_dtype = mpole_coeff.dtype
        if output_dtype is None:
            output_dtype = coeff_dtype

        from pytools.obj_array import make_obj_array
        outputs = make_obj_array([
            cl_array.empty(queue, target_count, output_dtype)
            for expr in self.output_maps])

        kernel = self.get_kernel(geometry_dtype, coeff_dtype, output_dtype)

        return outputs


