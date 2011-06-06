from __future__ import division

import numpy as np
from mako.template import Template
from pytools import memoize_method

import pyopencl as cl
import pyopencl.array as cl_array

from sumpy.kernel_common import (
        COMMON_PREAMBLE, FMMParameters)




P2M_KERNEL = Template(
COMMON_PREAMBLE +
"""//CL//
typedef ${offset_type} offset_t;
typedef ${mpole_offset_type} mpole_offset_t;
typedef ${coefficient_type} coeff_t;
typedef ${geometry_type} geometry_t;
typedef ${geometry_type}${dimensions} geometry_vec_t;


#define CELL_INDEX get_group_id(0)


// Work partition:
// ---------------
// Each work item is responsible for updating one expansion from a list of
// particles in a cell.
//
// FIXME: Should be one coefficient per work item, one cell per work group.

__kernel
void p2m(
    global const offset_t *cell_idx_to_particle_offset_g,
    global const uint *cell_idx_to_particle_cnt_g,
    % for i in range(dimensions):
        global const geometry_t *s${i}_g,
    % endfor
    global const coeff_t *restrict strength_g,
    % for i in range(dimensions):
        global const geometry_t *c${i}_g,
    % endfor
    global coeff_t *mpole_coeff_g)
{
    % for i in range(coeff_cnt):
        coeff_t mpole_coeff${i} = 0;
    % endfor

    uint particle_start = cell_idx_to_particle_cnt_g[CELL_INDEX];
    uint particle_cnt = cell_idx_to_particle_cnt_g[CELL_INDEX];

    ${load_vector("ctr", "c_g", "CELL_INDEX")}

    for (uint particle_idx = particle_start; 
        particle_idx < particle_start+particle_cnt; ++particle_idx)
    {
        geometry_vec_t src;
        ${load_vector("src", "s_g", "particle_idx")}

        coeff_t strength = strength_g[particle_idx];

        % for var_name, expr in vars_and_exprs:
            % if var_name.startswith("output"):
                ${var_name} += ${expr};
            % else:
                coeff_t ${var_name} = ${expr};
            % endif
        % endfor
    }

    % for i in range(coeff_cnt):
        mpole_coeff_g[CELL_INDEX * ${coef_cnt_padded} + ${i}] = mpole_coeff${i};
    % endfor
}
""", strict_undefined=True, disable_unicode=True)





class P2MKernel(object):
    def __init__(self, ctx, expansion,
            options=[], name="p2m", fmm_par=FMMParameters()):
        self.context = ctx
        self.expansion = expansion
        self.fmm_parameters = fmm_par

        self.options = options
        self.name = name

    @memoize_method
    def get_kernel(self, geometry_dtype, coeff_dtype, coeff_cnt_padded):
        dimensions = self.expansion.dimensions

        from sumpy.symbolic.codegen import generate_cl_statements_from_assignments
        from sumpy.symbolic import vector_subs, make_sym_vector

        old_var = make_sym_vector("a", dimensions)
        new_var = (make_sym_vector("c", dimensions)
                - make_sym_vector("s", dimensions))

        cs_coeffs = [
                vector_subs(basis_fun, old_var, new_var)
                for basis_fun in self.expansion.coefficients]

        outputs = [
                ("output%d" % output_idx,
                    output_map(sum(tc_basis_fun for tc_basis_fun in cs_coeffs)))

                for output_idx, output_map in enumerate(self.output_maps)
                ]

        from sumpy.symbolic.codegen import gen_c_source_subst_map
        subst_map = gen_c_source_subst_map(dimensions)

        vars_and_exprs = generate_cl_statements_from_assignments(
                outputs, subst_map=subst_map)

        from pyopencl.characterize import has_double_support
        from pyopencl.tools import dtype_to_ctype
        kernel_src = P2M_KERNEL.render(
                dimensions=dimensions,
                vars_and_exprs=vars_and_exprs,
                type_declarations="",
                coeff_cnt_padded=coeff_cnt_padded,
                double_support=all(
                    has_double_support(dev) for dev in self.context.devices),

                offset_type=dtype_to_ctype(self.fmm_parameters.offset_type),
                mpole_offset_type=dtype_to_ctype(self.fmm_parameters.mpole_offset_type),
                coefficient_type=dtype_to_ctype(coeff_dtype),
                geometry_type=dtype_to_ctype(geometry_dtype),
                )

        prg = cl.Program(self.context, kernel_src).build(self.options)
        return getattr(prg, self.name)

    def __call__(self,
            cell_idx_to_particle_offset, cell_idx_to_particle_cnt,
            sources, cell_centers, coeff_dtype,
            output_dtype=None, allocator=None, 
            queue=None, wait_for=None):

        # {{{ type processing

        geometry_dtype = sources[0].dtype
        coeff_dtype = np.dtype(coeff_dtype)

        coeff_cnt_padded = self.expansion.padded_coefficient_count(
                    coeff_dtype)

        # }}}

        queue = queue or sources[0].queue
        allocator = allocator or sources[0].allocator

        cell_count = len(cell_centers[0])

        mpole_coeff = np.empty((cell_count, coeff_cnt_padded),
                dtype=coeff_dtype)
        from pytools.obj_array import make_obj_array
        outputs = make_obj_array([
            cl_array.empty(queue, target_count, output_dtype,
                allocator=allocator)
            for expr in self.output_maps])

        tgt_cell_count = len(m2p_ilist_starts) - 1
        kernel = self.get_kernel(geometry_dtype, coeff_dtype, output_dtype,
                par_cell_cnt, coeff_cnt_padded)

        wg_size = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE)

        wg_dim = np.array([coeff_cnt_padded, par_cell_cnt])
        global_dim = wg_dim * np.array([tgt_cell_count, 1])

        kernel(queue, global_dim, wg_dim,
                *(
                    [tgt_i.data for tgt_i in targets]
                    + [out_i.data for out_i in outputs]
                    + [m2p_ilist_starts.data, m2p_ilist_mpole_offsets.data,
                        mpole_coeff.data, 
                        cell_idx_to_particle_offset.data,
                        cell_idx_to_particle_cnt.data]
                    ), wait_for=wait_for)

        return outputs


# vim: foldmethod=marker filetype=pyopencl.python
