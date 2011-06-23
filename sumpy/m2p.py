from __future__ import division

import numpy as np
import sympy as sp
from mako.template import Template
from pytools import memoize_method

import pyopencl as cl
import pyopencl.array as cl_array

from sumpy.kernel_common import (
        COMMON_PREAMBLE, FMMParameters)




M2P_KERNEL = Template(
COMMON_PREAMBLE +
r"""//CL//
typedef ${offset_type} offset_t;
typedef ${mpole_offset_type} mpole_offset_t;
typedef ${coefficient_type} coeff_t;
typedef ${geometry_type} geometry_t;
typedef ${geometry_type}${dimensions} geometry_vec_t;
typedef ${output_type} output_t;




#define TGT_CELL get_group_id(0)
<% wg_size = ctr_coeff_cnt_size*par_cell_cnt %>

// Work partition:
// ---------------
// Each work item is responsible for computing the summed-up multipole expansions
// at one target point. Each work group is responsible for one target cell.
// To achieve this, the kernel loops first over batches of particles within the target
// cell (if necessary) and then over source cells. The outer "loop over source
// cell batches" orchestrates cooperative loading of multipole coefficients.

__kernel
__attribute__((reqd_work_group_size(${ctr_coeff_cnt_size}, ${par_cell_cnt}, 1)))
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
    uint lid = get_local_id(0) + ${ctr_coeff_cnt_size} * get_local_id(1);

    offset_t ilist_start = m2p_ilist_starts_g[TGT_CELL];
    offset_t ilist_end = m2p_ilist_starts_g[TGT_CELL+1];

    offset_t tgt_cell_particle_offset = cell_idx_to_particle_offset_g[TGT_CELL];
    uint tgt_cell_particle_count = cell_idx_to_particle_cnt_g[TGT_CELL];

    __local coeff_t mpole_coeff_l[${par_cell_cnt} * ${ctr_coeff_cnt_size}];

    // index into this cell's list of particles
    uint plist_base = 0;

    // loop over particle batches

    // NOTE: Since there is barrier synchronization at the innermost 
    // level of the loop, all work items must participate until the
    // end of the loop, whether they need to or not.

    // TODO: Convert to batched loop
    while (plist_base < tgt_cell_particle_count)
    {
        uint plist_idx = plist_base+lid;
        geometry_vec_t tgt;

        if (plist_idx < tgt_cell_particle_count)
        {
            ${load_vector_g("tgt", "t", "tgt_cell_particle_offset + plist_idx")}
        }

        % for i in range(output_count):
            output_t output${i} = 0;
        % endfor

        offset_t ilist_base = ilist_start;

        // loop over source cell batches
        // TODO: Convert to batched loop
        while (ilist_base < ilist_end)
        {
            // index into overall M2P interaction list
            offset_t ilist_idx = ilist_base + get_local_id(1);

            barrier(CLK_LOCAL_MEM_FENCE);
            if (ilist_idx < ilist_end)
            {
                mpole_offset_t mpole_offset = m2p_ilist_mpole_offsets_g[ilist_idx];
                mpole_coeff_l[lid] = mpole_coeff_g[mpole_offset + get_local_id(0)];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            uint batch_cell_cnt = min(${par_cell_cnt}, ilist_end-ilist_base);

            // loop over source cells
            for (uint src_cell = 0; src_cell < batch_cell_cnt; ++src_cell)
            {
                uint loc_mpole_base = src_cell*${ctr_coeff_cnt_size};
                geometry_vec_t ctr;
                % for i in range(dimensions):
                    ctr.s${i} = mpole_coeff_l[loc_mpole_base+${i}];
                % endfor
                loc_mpole_base += ${dimensions};

                #define COEFF(i) mpole_coeff_l[loc_mpole_base+${dimensions}+i]

                % for var, expr in vars_and_exprs:
                    % if var.startswith("output"):
                        ${var} += ${expr};
                    % else:
                        output_t ${var} = ${expr};
                    % endif
                % endfor
            }

            ilist_base += ${par_cell_cnt};
        }

        if (plist_idx < tgt_cell_particle_count)
        {
            % for i in range(output_count):
              output${i}_g[tgt_cell_particle_offset + plist_idx] = output${i};
            % endfor
        }

        plist_base += ${wg_size};
    }
}
""", strict_undefined=True, disable_unicode=True)




class M2PKernel(object):
    def __init__(self, ctx, expansion, output_maps=[lambda x: x],
            options=[], name="m2p", fmm_par=FMMParameters()):
        """
        :output_maps: A list of functions which will be applied to basis functions
          in the expansion, each generating a different output array.
        """
        self.context = ctx
        self.expansion = expansion
        self.output_maps = output_maps
        self.fmm_parameters = fmm_par

        self.options = options
        self.name = name

        self.max_wg_size = min(dev.max_work_group_size for dev in ctx.devices)

    @memoize_method
    def get_kernel(self, geometry_dtype, coeff_dtype, output_dtype, 
            par_cell_cnt, ctr_coeff_cnt_size):
        dimensions = self.expansion.dimensions

        from sumpy.symbolic.codegen import generate_cl_statements_from_assignments
        from sumpy.symbolic import vector_subs, make_sym_vector

        old_var = make_sym_vector("b", dimensions)
        new_var = (make_sym_vector("t", dimensions)
                - make_sym_vector("c", dimensions))

        tc_basis = [
                vector_subs(basis_fun, old_var, new_var)
                for basis_fun in self.expansion.basis]

        outputs = [
                ("output%d" % output_idx,
                    output_map(sum(
                        sp.Symbol("COEFF(%d)" % i)*tc_basis_fun
                        for i, tc_basis_fun in enumerate(tc_basis))))

                for output_idx, output_map in enumerate(self.output_maps)
                ]

        from sumpy.symbolic.codegen import gen_c_source_subst_map
        subst_map = gen_c_source_subst_map(dimensions)

        vars_and_exprs = generate_cl_statements_from_assignments(
                outputs, subst_map=subst_map)

        from pyopencl.characterize import has_double_support
        from pyopencl.tools import dtype_to_ctype
        kernel_src = M2P_KERNEL.render(
                dimensions=dimensions,
                vars_and_exprs=vars_and_exprs,
                type_declarations="",
                par_cell_cnt=par_cell_cnt,
                ctr_coeff_cnt_size=ctr_coeff_cnt_size,
                output_count=len(self.output_maps),
                double_support=all(
                    has_double_support(dev) for dev in self.context.devices),

                offset_type=dtype_to_ctype(self.fmm_parameters.offset_type),
                mpole_offset_type=dtype_to_ctype(self.fmm_parameters.mpole_offset_type),
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
            output_dtype=None, allocator=None, 
            queue=None, wait_for=None):
        target_count, = targets[0].shape

        # {{{ type processing

        geometry_dtype = targets[0].dtype
        coeff_dtype = np.dtype(mpole_coeff.dtype)
        if output_dtype is None:
            output_dtype = coeff_dtype

        ctr_coeff_cnt_size = self.expansion.padded_coefficient_count_with_center(
                    coeff_dtype)

        # }}}

        # {{{ determine work group size

        wg_size = min(self.max_wg_size, 256) # FIXME: Tune?
        from pytools import div_ceil
        par_cell_cnt = div_ceil(wg_size, ctr_coeff_cnt_size)
        wg_size = par_cell_cnt * ctr_coeff_cnt_size

        # }}}

        queue = queue or targets[0].queue
        allocator = allocator or targets[0].allocator

        from pytools.obj_array import make_obj_array
        outputs = make_obj_array([
            cl_array.empty(queue, target_count, output_dtype,
                allocator=allocator)
            for expr in self.output_maps])

        tgt_cell_count = len(m2p_ilist_starts) - 1
        kernel = self.get_kernel(geometry_dtype, coeff_dtype, output_dtype,
                par_cell_cnt, ctr_coeff_cnt_size)

        kernel(queue, (tgt_cell_count, 1), (ctr_coeff_cnt_size, par_cell_cnt),
                *(
                    [tgt_i.data for tgt_i in targets]
                    + [out_i.data for out_i in outputs]
                    + [m2p_ilist_starts.data, m2p_ilist_mpole_offsets.data,
                        mpole_coeff.data, 
                        cell_idx_to_particle_offset.data,
                        cell_idx_to_particle_cnt.data]
                    ), wait_for=wait_for, g_times_l=True)

        return outputs




# vim: foldmethod=marker filetype=pyopencl
