import numpy as np




class FMMParameters:
    offset_type = np.uint32
    mpole_offset_type = np.uint32




COMMON_PREAMBLE = r"""//CL//
% if double_support:
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
% endif
## #pragma OPENCL EXTENSION cl_amd_printf: enable

<%def name="load_vector_g(tgt_name, src_name, src_base_offset)">
  % for i in range(dimensions):
    ${tgt_name}.s${i} = ${src_name}${i}_g[${src_base_offset}];
  % endfor
</%def>

<%def name="load_vector_l(tgt_name, src_name, src_base_offset)">
  % for i in range(dimensions):
    ${tgt_name}.s${i} = ${src_name}${i}_l[${src_base_offset}];
  % endfor
</%def>


<%def name="chunk_for_with_tail(loop_var, start, chunk_size, end,
    loop_var_type='uint', debug=False)">
% if not debug:
    {
      ${loop_var_type} ${loop_var} = ${start};
      while (${loop_var} + ${chunk_size} < ${end})
      {
        ${caller.body(is_tail=False, chunk_length=chunk_size)}

        ${loop_var} += ${chunk_size};
      }
      ${caller.body(is_tail=True, chunk_length="%s-%s" % (end, loop_var))}
    }
% else:
    for (${loop_var_type} ${loop_var} = ${start}; ${loop_var} < ${end}; 
      ${loop_var} += ${chunk_size})
    {
        ${caller.body(is_tail=True, 
            chunk_length="min(%s, %s-%s)" % (chunk_size, end, loop_var))}
    }

% endif
</%def>
"""



# vim: foldmethod=marker filetype=pyopencl.python
