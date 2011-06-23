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


#define CELL_INDEX get_global_id(0)


// Work partition:
// ---------------
// Each work item is responsible for updating one expansion from a list of
// particles in a cell.
//
// FIXME: Should be one coefficient per work item, one cell per work group.

__kernel
void p2m(
    unsigned cell_count,
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
    if (CELL_INDEX >= cell_count)
        return;

    % for i in range(coeff_cnt):
        coeff_t mpole_coeff${i} = 0;
    % endfor

    uint particle_start = cell_idx_to_particle_offset_g[CELL_INDEX];
    uint particle_cnt = cell_idx_to_particle_cnt_g[CELL_INDEX];

    geometry_vec_t ctr;
    ${load_vector_g("ctr", "c", "CELL_INDEX")}

    for (uint particle_idx = particle_start; 
        particle_idx < particle_start+particle_cnt; ++particle_idx)
    {
        geometry_vec_t src;
        ${load_vector_g("src", "s", "particle_idx")}

        coeff_t strength = strength_g[particle_idx];

        % for var_name, expr in vars_and_exprs:
            % if var_name.startswith("mpole_coeff"):
                ${var_name} += ${expr};
            % else:
                coeff_t ${var_name} = strength * ${expr};
            % endif
        % endfor
    }

    % for i in range(dimensions):
        mpole_coeff_g[CELL_INDEX * ${coeff_cnt_padded} + ${i}] = ctr.s${i};
    % endfor
    % for i in range(coeff_cnt):
        mpole_coeff_g[CELL_INDEX * ${coeff_cnt_padded} + ${dimensions} + ${i}]
          = mpole_coeff${i};
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
                ("mpole_coeff%d" % coeff_idx, cs_coeff)
                for coeff_idx, cs_coeff in enumerate(cs_coeffs)
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
                coeff_cnt=len(self.expansion.coefficients),
                coeff_cnt_padded=coeff_cnt_padded,
                double_support=all(
                    has_double_support(dev) for dev in self.context.devices),

                offset_type=dtype_to_ctype(self.fmm_parameters.offset_type),
                mpole_offset_type=dtype_to_ctype(self.fmm_parameters.mpole_offset_type),
                coefficient_type=dtype_to_ctype(coeff_dtype),
                geometry_type=dtype_to_ctype(geometry_dtype),
                )

        prg = cl.Program(self.context, kernel_src).build(self.options)
        knl = getattr(prg, self.name)
        knl.set_scalar_arg_dtypes([np.uint32] + [None]*(4+2*dimensions))
        return knl

    def __call__(self,
            cell_idx_to_particle_offset, cell_idx_to_particle_cnt,
            sources, strength, cell_centers, coeff_dtype,
            output_dtype=None, allocator=None,
            queue=None, wait_for=None):

        # {{{ type processing

        geometry_dtype = sources[0].dtype
        coeff_dtype = np.dtype(coeff_dtype)

        coeff_cnt_padded = self.expansion.padded_coefficient_count_with_center(
                    coeff_dtype)

        # }}}

        queue = queue or sources[0].queue
        allocator = allocator or sources[0].allocator

        cell_count = len(cell_centers[0])

        mpole_coeff = cl_array.empty(queue, (cell_count, coeff_cnt_padded),
                dtype=coeff_dtype)

        kernel = self.get_kernel(geometry_dtype, coeff_dtype, coeff_cnt_padded)

        wg_size = kernel.get_work_group_info(
                cl.kernel_work_group_info.WORK_GROUP_SIZE,
                queue.device)

        from pytools import div_ceil
        wg_count = div_ceil(cell_count, wg_size)

        kernel(queue, (wg_count,), (wg_size,),
                *(
                    [cell_count]
                    + [cell_idx_to_particle_offset.data,
                        cell_idx_to_particle_cnt.data]
                    + [src_i.data for src_i in sources]
                    + [strength.data]
                    + [ctr_i.data for ctr_i in cell_centers]
                    + [mpole_coeff.data]
                    ), wait_for=wait_for, g_times_l=True)

        return mpole_coeff




# vim: foldmethod=marker filetype=pyopencl
