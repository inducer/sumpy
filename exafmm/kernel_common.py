COMMON_PREAMBLE = """
% if double_support:
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
% endif

<%def name="load_vector(tgt_name, src_name, src_base_offset)">
  % for i in range(dimensions):
    ${tgt_name}.s${i} = ${src_name}${i}_g[${src_base_offset}];
  % endfor
</%def>
"""
