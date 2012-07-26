from __future__ import division
import numpy as np
from pytools import memoize, memoize_method, Record




class _KernelInfo(Record):
    pass




# Note that hbound must be slightly larger than the highest found coordinate,
# so that 1.0 is never reached as a scaled coordinate.

MORTON_CODEGEN = """//CL//
<%def name="morton(output_var, start_bit, levels, coords, bounds, tgt_bits=32)">
    {
        %for i, (coord, (lbound, hbound)) in enumerate(zip(coords, bounds)):
            unsigned coord_${i} = (unsigned) (((coord-lbound)/(hbound-lbound)) * (1U << (${start_bit} + ${levels})));
        %endfor

        <%

        bit_count = [tgt_bits]

        def get_next_bit():
            bit_count[0] -= 1
            return bit_count[0]

        %>

        ${output_var} = 0;
        ${output_var} = ${output_var}
        %for l in range(levels):
            %for i in range(len(coords)):
                <%
                  at_bit = levels-l-1 # numbering from 0
                  want_bit = get_next_bit()
                %>
                | (coord_${i} & (1U << ${at_bit})) << ${want_bit-at_bit}
            %endfor
        %endfor
            ;
    }

</%def>

<%
cnames = ["x", "y", "z"]
%>
${morton("mvar", 0, 4, cnames, [("%sl" % cn, "%sh" % cn) for cn in cnames])}
"""



TREE_BUILD_SCAN_UTILS = """//CL//

"""





AXIS_NAMES = ["x", "y", "z", "w"]

# {{{ bounding box finding

@memoize
def make_bounding_box_dtype(dimensions, geo_dtype):
    fields = []
    for i in range(dimensions):
        fields.append(("min_%s" % AXIS_NAMES[i], geo_dtype))
        fields.append(("max_%s" % AXIS_NAMES[i], geo_dtype))

    dtype = np.dtype(fields)

    from pyopencl.tools import register_dtype
    register_dtype(dtype, "sumpy_bbox_%dd_t" % dimensions)
    return dtype

BBOX_CODE_TEMPLATE = r"""//CL//
    ${bbox_struct_decl}

    typedef sumpy_bbox_${dimensions}d_t bbox_t;
    typedef ${geo_ctype} coord_t;

    bbox_t bbox_neutral()
    {
        bbox_t result;
        %for ax in axis_names:
            result.min_${ax} = ${geo_dtype_3ltr}_MAX;
            result.max_${ax} = -${geo_dtype_3ltr}_MAX;
        %endfor
        return result;
    }

    bbox_t bbox_from_point(${", ".join("coord_t %s" % ax for ax in axis_names)})
    {
        bbox_t result;
        %for ax in axis_names:
            result.min_${ax} = ${ax};
            result.max_${ax} = ${ax};
        %endfor
        return result;
    }

    bbox_t agg_bbox(bbox_t a, bbox_t b)
    {
        %for ax in axis_names:
            a.min_${ax} = min(a.min_${ax}, b.min_${ax});
            a.max_${ax} = max(a.max_${ax}, b.max_${ax});
        %endfor
        return a;
    }
"""

class BoundingBoxFinder:
    def __init__(self, context):
        self.context = context

    @memoize_method
    def get_kernel(self, dimensions, geo_dtype):
        from pyopencl.tools import dtype_to_c_struct, dtype_to_ctype
        bbox_dtype = make_bounding_box_dtype(dimensions, geo_dtype)

        if geo_dtype == np.float64:
            geo_dtype_3ltr = "DBL"
        elif geo_dtype == np.float32:
            geo_dtype_3ltr = "FLT"
        else:
            raise TypeError("unknown geo_dtype")

        axis_names = AXIS_NAMES[:dimensions]

        geo_ctype = dtype_to_ctype(geo_dtype)

        from mako.template import Template
        preamble = Template(BBOX_CODE_TEMPLATE, strict_undefined=True).render(
                axis_names=axis_names,
                dimensions=dimensions,
                geo_ctype=dtype_to_ctype(geo_dtype),
                geo_dtype_3ltr=geo_dtype_3ltr,
                bbox_struct_decl=dtype_to_c_struct(bbox_dtype)
                )

        from pyopencl.reduction import ReductionKernel
        return ReductionKernel(self.context, bbox_dtype,
                neutral="bbox_neutral()",
                reduce_expr="agg_bbox(a, b)",
                map_expr="bbox_from_point(%s)" % ", ".join(
                    "%s[i]" % ax for ax in axis_names),
                arguments=", ".join(
                    "__global %s *%s" % (geo_ctype, ax) for ax in axis_names),
                preamble=preamble)

    def __call__(self, points):
        dimensions = len(points)

        from pytools import single_valued
        geo_dtype = single_valued(coord.dtype for coord in points)

        return self.get_kernel(dimensions, geo_dtype)(*points)

# }}}




def padded_bin(i, l):
    s = bin(i)[2:]
    while len(s) < l:
        s = '0' + s
    return s

@memoize
def make_scan_type(dimensions):
    fields = [
            ('box_id', np.uint32),
            ('parent', np.uint32),
            ('count', np.uint32),
            ]

    for i in range(2**dimensions):
        fields.append(('count_%s' % padded_bin(i, dimensions), np.uint32))

    dtype = np.dtype(fields)
    from pyopencl.tools import register_dtype
    register_dtype(dtype, "sumpy_tree_scan_%dd_t" % dimensions)
    return dtype






class Tree(Record):
    pass




class TreeBuilder(object):
    def __init__(self, context):
        self.context = context
        self.bbox_finder = BoundingBoxFinder(context)

    @memoize_method
    def get_built_kernels(self, dimensions, geo_dtype):
        from pyopencl.tools import dtype_to_c_struct, dtype_to_ctype
        geo_ctype = dtype_to_ctype(geo_dtype)
        scan_dtype = make_scan_type(dimensions)
        scan_ctype = dtype_to_ctype(scan_dtype)

        scan_type_decl = dtype_to_c_struct(scan_dtype)

        preamble = (scan_type_decl 
                + "\n\ntypedef sutree_scan_%dd_t tree_scan_t;\n" % dimensions)

        from pyopencl.elementwise import ElementwiseKernel
        #to_scan_map = ElementwiseKernel(
                #", ".join(["tree_scan_t *out", "%s *xcoords" %



        return _KernelInfo(
                scan_dtype=scan_dtype,
                )




    def __call__(self, queue, points):
        bbox = self.bbox_finder(points).get()
        print bbox
        1/0

        dimensions = len(points)

        from pytools import single_valued
        geo_dtype = single_valued(coord.dtype for coord in points)
        scan_info = self.get_scan_kernel(dimensions, geo_dtype)

        nparticles = single_valued(len(coord) for coord in points)







def build_tree(points):
    from mako.template import Template
    print Template(MORTON_CODEGEN, strict_undefined=True).render()




# vim: filetype=pyopencl:fdm=marker
