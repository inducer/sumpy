from __future__ import division

import numpy as np
import sympy as sp
from mako.template import Template




M2P_KERNEL = Template("""
${type_declarations}

<%def name="load_vector(tgt_name, src_name, src_base_offset)">
  % for i in range(dimensions):
    ${tgt_name}.s${i} = ${src_name}${i}_g[${src_base_offset}];
  % endfor
</%def>

#define TGT_CELL get_group_id(0)
<% wg_size = coef_cnt_padded*par_cell_cnt %>

__kernel
__attribute__((reqd_work_group_size(${coef_cnt_padded}, ${par_cell_cnt}, 1)))
void m2p(
% for i in range(dimensions):
  const geometry_t *c${i}_g,
% endfor
% for i in range(dimensions):
  const geometry_t *t${i}_g,
% endfor
% for i in range(output_count):
  output_t *output${i}_g,
% endfor
  const offset_t *m2l_ilist_starts_g,
  const mpole_offset_t *m2l_ilist_mpole_offsets_g,
  const coeff_t *mpole_coeff_g,
  const offset_t *cell_idx_to_particle_offset_g,
  const uint32_t *cell_idx_to_particle_cnt_g
  )

{
  uint32_t lid = get_local_id(0) + ${coef_cnt_padded} * get_local_id(1);

  offset_t ilist_start = m2l_ilist_starts[TGT_CELL];
  offset_t ilist_end = m2l_ilist_starts[TGT_CELL+1];

  offset_t tgt_cell_particle_offset = cell_idx_to_particle_offset_g[TGT_CELL];
  uint32_t tgt_cell_particle_count = cell_idx_to_particle_cnt_g[TGT_CELL];

  __local coeff_t mpole_coeff_l[${par_cell_cnt} * ${coef_cnt_padded}];

  // index into this cell's list of particles
  uint32_t plist_idx = lid;

  // loop over particle batches
  while (plist_idx < tgt_cell_particle_count)
  {
    geometry_vec_t tgt;
    ${load_vector("tgt", "t", "tgt_cell_particle_offset + plist_idx")}

    % for i in range(output_count):
      output_t output${i} = 0;
    % endfor

    // index into overall M2L interaction list
    offset_t ilist_idx = ilist_start + get_local_id(1);

    // loop over source cell batches
    while (ilist_idx < ilist_end)
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      mpole_offset_t mpole_offset = m2l_ilist_mpole_offsets_g[ilist_idx];
      mpole_coeff_l[lid] = mpole_coeff_g[mpole_offset + get_local_id(0)];
      barrier(CLK_LOCAL_MEM_FENCE);

      for (uint32_t src_cell = 0; src_cell < ${par_cell_cnt}; ++src_cell)
      {
        uint32_t loc_mpole_base = src_cell*${coef_cnt_padded};
        geometry_vec_t ctr;
        % for i in range(dimensions):
          ctr.s${i} = mpole_coeff_l[loc_mpole_base+${i}];
        % endfor
        loc_mpole_base += ${dimensions};

        % for var, expr in vars_and_exprs:
          % if var.startswith("output"):
            ${var} += ${expr};
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
""")






def make_m2p_source(coeff_dtype, expansion, output_maps=[lambda x: x]):
    coeff_dtype = np.dtype(coeff_dtype)

    dimensions = expansion.dimensions

    from exafmm.symbolic.codegen import generate_cl_statements_from_assignments
    from exafmm.symbolic import vector_subs, make_sym_vector

    old_var = make_sym_vector("b", dimensions)
    new_var = (make_sym_vector("t", dimensions)
            - make_sym_vector("c", dimensions))

    tc_basis = [
            vector_subs(basis_fun, old_var, new_var)
            for basis_fun in expansion.basis]

    outputs = [
            ("output%d" % output_idx,
                output_map(tc_basis_fun))

            for output_idx, output_map in enumerate(output_maps)
            for tc_basis_fun in tc_basis]

    from exafmm.symbolic.codegen import gen_c_source_subst_map
    subst_map = gen_c_source_subst_map(dimensions)

    vars_and_exprs = generate_cl_statements_from_assignments(
            outputs, subst_map=subst_map)

    return M2P_KERNEL.render(
            dimensions=dimensions,
            vars_and_exprs=vars_and_exprs,
            type_declarations="",
            par_cell_cnt=10,
            coef_cnt_padded=expansion.padded_coefficient_count(
                coeff_dtype),
            output_count=len(output_maps),
            )
