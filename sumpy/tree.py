from __future__ import division
import numpy as np
from pytools import memoize, memoize_method, Record
import pyopencl as cl
import pyopencl.array
from mako.template import Template




AXIS_NAMES = ["x", "y", "z", "w"]

# {{{ bounding box finding

@memoize
def make_bounding_box_dtype(device, dimensions, geo_dtype):
    fields = []
    for i in range(dimensions):
        fields.append(("min_%s" % AXIS_NAMES[i], geo_dtype))
        fields.append(("max_%s" % AXIS_NAMES[i], geo_dtype))

    dtype = np.dtype(fields)

    name = "sumpy_bbox_%dd_t" % dimensions

    from pyopencl.tools import register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)
    register_dtype(dtype, name)

    return dtype, c_decl




BBOX_CODE_TEMPLATE = Template(r"""//CL//
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

    bbox_t bbox_from_particle(${", ".join("coord_t %s" % ax for ax in axis_names)})
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
""", strict_undefined=True)

class BoundingBoxFinder:
    def __init__(self, context):
        self.context = context

    @memoize_method
    def get_kernel(self, dimensions, geo_dtype):
        from pyopencl.tools import dtype_to_ctype
        bbox_dtype, bbox_cdecl = make_bounding_box_dtype(
                self.context.devices[0], dimensions, geo_dtype)

        if geo_dtype == np.float64:
            geo_dtype_3ltr = "DBL"
        elif geo_dtype == np.float32:
            geo_dtype_3ltr = "FLT"
        else:
            raise TypeError("unknown geo_dtype")

        axis_names = AXIS_NAMES[:dimensions]

        geo_ctype = dtype_to_ctype(geo_dtype)

        preamble = BBOX_CODE_TEMPLATE.render(
                axis_names=axis_names,
                dimensions=dimensions,
                geo_ctype=dtype_to_ctype(geo_dtype),
                geo_dtype_3ltr=geo_dtype_3ltr,
                bbox_struct_decl=bbox_cdecl
                )

        from pyopencl.reduction import ReductionKernel
        return ReductionKernel(self.context, bbox_dtype,
                neutral="bbox_neutral()",
                reduce_expr="agg_bbox(a, b)",
                map_expr="bbox_from_particle(%s)" % ", ".join(
                    "%s[i]" % ax for ax in axis_names),
                arguments=", ".join(
                    "__global %s *%s" % (geo_ctype, ax) for ax in axis_names),
                preamble=preamble)

    def __call__(self, particles):
        dimensions = len(particles)

        from pytools import single_valued
        geo_dtype = single_valued(coord.dtype for coord in particles)

        return self.get_kernel(dimensions, geo_dtype)(*particles)

# }}}

class _KernelInfo(Record):
    pass

def padded_bin(i, l):
    s = bin(i)[2:]
    while len(s) < l:
        s = '0' + s
    return s

# {{{ data types

@memoize
def make_morton_bin_count_type(device, dimensions, particle_id_dtype):
    fields = []
    for mnr in range(2**dimensions):
        fields.append(('c%s' % padded_bin(mnr, dimensions), particle_id_dtype))

    dtype = np.dtype(fields)

    name = "sumpy_morton_bin_count_%dd_t" % dimensions
    from pyopencl.tools import register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)

    # FIXME: build id_type into name
    register_dtype(dtype, name)
    return dtype, c_decl

@memoize
def make_scan_type(device, dimensions, particle_id_dtype, box_id_dtype):
    morton_dtype, _ = make_morton_bin_count_type(device, dimensions, particle_id_dtype)
    dtype = np.dtype([
            ('counts', morton_dtype),
            ('current_box_id', box_id_dtype), # max-scanned
            ('subdivided_box_id', box_id_dtype), # sum-scanned
            ('morton_nr', np.uint8),
            ])

    name = "sumpy_tree_scan_%dd_t" % dimensions
    from pyopencl.tools import register_dtype, match_dtype_to_c_struct
    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)

    # FIXME: build id_types into name
    register_dtype(dtype, name)
    return dtype, c_decl

# }}}




# {{{ preambles

PREAMBLE_TPL = Template(r"""//CL//
    ${bbox_type_decl}
    ${morton_bin_count_type_decl}
    ${tree_scan_type_decl}

    typedef sumpy_morton_bin_count_${dimensions}d_t morton_t;
    typedef sumpy_tree_scan_${dimensions}d_t scan_t;
    typedef sumpy_bbox_${dimensions}d_t bbox_t;
    typedef ${geo_ctype} coord_t;
    typedef ${box_id_ctype} box_id_t;
    typedef ${particle_id_ctype} particle_id_t;


    <%
      def get_count_branch(known_bits):
          if len(known_bits) == dimensions:
              return "counts.c%s" % known_bits

          dim = len(known_bits)
          boundary_morton_nr = known_bits + "1" + (dimensions-dim-1)*"0"

          return ("((morton_nr < %s) ? %s : %s)" % (
              int(boundary_morton_nr, 2),
              get_count_branch(known_bits+"0"),
              get_count_branch(known_bits+"1")))
    %>

    int get_count(morton_t counts, int morton_nr)
    {
        return ${get_count_branch("")};
    }

    #ifdef DEBUG
        #define dbg_printf(ARGS) printf ARGS
    #else
        #define dbg_printf(ARGS) /* */
    #endif

""", strict_undefined=True)

# }}}

# {{{ scan primitive code template

SCAN_PREAMBLE_TPL = Template(r"""//CL//
    scan_t scan_t_neutral()
    {
        scan_t result;
        %for mnr in range(2**dimensions):
            result.counts.c${padded_bin(mnr, dimensions)} = 0;
        %endfor
        result.current_box_id = 0;
        result.subdivided_box_id = 0;
        return result;
    }

    scan_t scan_t_add(scan_t a, scan_t b, bool across_seg_boundary)
    {
        if (!across_seg_boundary)
        {
            %for mnr in range(2**dimensions):
                <% field = "counts.c"+padded_bin(mnr, dimensions) %>
                b.${field} = a.${field} + b.${field};
            %endfor
            b.current_box_id = max(a.current_box_id, b.current_box_id);
        }

        // subdivided_box_id must use a non-segmented scan to globally
        // assign box numbers.
        b.subdivided_box_id = a.subdivided_box_id + b.subdivided_box_id;

        // directly use b.morton_nr
        return b;
    }

    scan_t scan_t_from_particle(
        int i,
        int level,
        box_id_t box_id,
        box_id_t box_count,
        particle_id_t box_start,
        particle_id_t box_particle_count,
        particle_id_t max_particles_in_box,
        bbox_t *bbox
        %for ax in axis_names:
            , coord_t ${ax}
        %endfor
    )
    {
        // Note that upper bound must be slightly larger than the highest found coordinate,
        // so that 1.0 is never reached as a scaled coordinate.

        %for ax in axis_names:
            unsigned ${ax}_bits = (unsigned) (
                ((${ax}-bbox->min_${ax})/(bbox->max_${ax}-bbox->min_${ax}))
                * (1U << (level+1)));
        %endfor

        unsigned level_morton_number = 0
        %for iax, ax in enumerate(axis_names):
            | (${ax}_bits & 1U) << (${dimensions-1-iax})
        %endfor
            ;

        scan_t result;
        %for mnr in range(2**dimensions):
            <% field = "counts.c"+padded_bin(mnr, dimensions) %>
            result.${field} = (level_morton_number == ${mnr});
        %endfor
        result.morton_nr = level_morton_number;

        /* my_box_id only valid for box starts at this particle, but that's ok.
         * We'll scan the box id (by max) so by output time every particle
         * knows its (by then possibly former) box id. */

        box_id_t my_box_id = box_id;
        result.current_box_id = my_box_id;

        /* subdivided_box_id is not very meaningful now, but when scanned over
         * by addition, will yield ids for boxes that are created by
         * subdividing the current (over-full) box.
         */
        result.subdivided_box_id = 0;
        if (i == 0)
            result.subdivided_box_id = box_count;
        if (i == box_start
            && box_particle_count > max_particles_in_box)
        {
            /* Grab ids for all subboxes at the start of the box.
             * Subboxes will have to subtract to find their id.
             */
            result.subdivided_box_id += ${2**dimensions};
        }

        return result;
    }

""", strict_undefined=True)

# }}}

# {{{ scan output code template

SCAN_OUTPUT_STMT_TPL = Template(r"""//CL//
    {
        particle_id_t my_id_in_my_box = -1
        %for mnr in range(2**dimensions):
            + item.counts.c${padded_bin(mnr, dimensions)}
        %endfor
            ;
        dbg_printf(("my_id_in_my_box:%d\n", my_id_in_my_box));
        morton_bin_counts[i] = item.counts;
        morton_nrs[i] = item.morton_nr;

        particle_id_t box_particle_count = box_particle_counts[item.current_box_id];

        unsplit_box_ids[i] = item.current_box_id;
        split_box_ids[i] = item.subdivided_box_id;

        /* Am I the last particle in my current box? */
        if (my_id_in_my_box+1 == box_particle_count)
        {
            dbg_printf(("store box %d cbi:%d\n", i, item.current_box_id));
            dbg_printf(("   store_sums: %d %d %d %d\n", item.counts.c00, item.counts.c01, item.counts.c10, item.counts.c11));
            box_morton_bin_counts[item.current_box_id] = item.counts;
        }

        /* Am I the last particle overall? If so, write box count. */
        if (i+1 == N)
            *box_count = item.subdivided_box_id;
    }
""", strict_undefined=True)

# }}}

# {{{ postprocessing kernel

POSTPROC_KERNEL_TPL =  Template(r"""//CL//
    morton_t my_morton_bin_counts = morton_bin_counts[i];
    box_id_t my_box_id = unsplit_box_ids[i];

    dbg_printf(("postproc %d:\n", i));
    dbg_printf(("   my_sums: %d %d %d %d\n",
        my_morton_bin_counts.c00, my_morton_bin_counts.c01,
        my_morton_bin_counts.c10, my_morton_bin_counts.c11));
    dbg_printf(("   my box id: %d\n", my_box_id));

    particle_id_t box_particle_count = box_particle_counts[my_box_id];

    /* Is this box being split? */
    if (box_particle_count > max_particles_in_box)
    {
        unsigned char my_morton_nr = morton_nrs[i];
        dbg_printf(("   my morton nr: %d\n", my_morton_nr));

        box_id_t new_box_id = split_box_ids[i] - ${2**dimensions} + my_morton_nr;
        dbg_printf(("   new_box_id: %d\n", new_box_id));

        morton_t my_box_morton_bin_counts = box_morton_bin_counts[my_box_id];
        /*
        dbg_printf(("   box_sums: %d %d %d %d\n", my_box_morton_bin_counts.c00, my_box_morton_bin_counts.c01, my_box_morton_bin_counts.c10, my_box_morton_bin_counts.c11));
        */

        particle_id_t my_count = get_count(my_morton_bin_counts, my_morton_nr);

        particle_id_t my_box_start = box_starts[my_box_id];
        particle_id_t tgt_particle_idx = my_box_start + my_count-1;
        %for mnr in range(2**dimensions):
            <% bin_nmr = padded_bin(mnr, dimensions) %>
            tgt_particle_idx += 
                (my_morton_nr > ${mnr}) 
                    ? my_box_morton_bin_counts.c${bin_nmr}
                    : 0;
        %endfor

        dbg_printf(("   moving %d -> %d\n", i, tgt_particle_idx));
        %for ax in axis_names:
            sorted_${ax}[tgt_particle_idx] = ${ax}[i];
        %endfor

        box_ids[tgt_particle_idx] = new_box_id;

        %for mnr in range(2**dimensions):
          /* Am I the last particle in my Morton bin? */
            if (${mnr} == my_morton_nr
                && my_box_morton_bin_counts.c${padded_bin(mnr, dimensions)} == my_count)
            {
                dbg_printf(("   ## splitting\n"));

                particle_id_t new_box_start = my_box_start
                %for sub_mnr in range(mnr):
                    + my_box_morton_bin_counts.c${padded_bin(sub_mnr, dimensions)}
                %endfor
                    ;

                dbg_printf(("   new_box_start: %d\n", new_box_start));

                box_start_flags[new_box_start] = 1;
                box_starts[new_box_id] = new_box_start;
                parent_ids[new_box_id] = my_box_id;

                box_particle_counts[new_box_id] = 
                    my_box_morton_bin_counts.c${padded_bin(mnr, dimensions)};

                dbg_printf(("   box pcount: %d\n", box_particle_counts[new_box_id]));
            }
        %endfor
    }
""", strict_undefined=True)

# }}}




class Tree(Record):
    pass



# {{{ driver

class TreeBuilder(object):
    def __init__(self, context, options=[]):
        self.context = context
        self.bbox_finder = BoundingBoxFinder(context)
        self.options = options

    # {{{ kernel creation

    @memoize_method
    def get_kernel_info(self, dimensions, geo_dtype,
            particle_id_dtype=np.uint32, box_id_dtype=np.uint32):

        from pyopencl.tools import dtype_to_c_struct, dtype_to_ctype
        geo_ctype = dtype_to_ctype(geo_dtype)

        particle_id_dtype = np.dtype(particle_id_dtype)
        particle_id_ctype = dtype_to_ctype(particle_id_dtype)

        box_id_dtype = np.dtype(box_id_dtype)
        box_id_ctype = dtype_to_ctype(box_id_dtype)

        dev = self.context.devices[0]
        scan_dtype, scan_type_decl = make_scan_type(dev,
                dimensions, particle_id_dtype, box_id_dtype)
        morton_bin_count_dtype, _ = scan_dtype.fields["counts"]
        bbox_dtype, bbox_type_decl = make_bounding_box_dtype(
                dev, dimensions, geo_dtype)

        axis_names = AXIS_NAMES[:dimensions]

        codegen_args = dict(
                dimensions=dimensions,
                axis_names=axis_names,
                padded_bin=padded_bin,
                geo_ctype=geo_ctype,
                morton_bin_count_type_decl=dtype_to_c_struct(
                    dev, morton_bin_count_dtype),
                tree_scan_type_decl=scan_type_decl,
                bbox_type_decl=dtype_to_c_struct(dev, bbox_dtype),
                particle_id_ctype=particle_id_ctype,
                box_id_ctype=box_id_ctype,
                )

        preamble = PREAMBLE_TPL.render(**codegen_args)
        scan_preamble = preamble + SCAN_PREAMBLE_TPL.render(**codegen_args)

        from pyopencl.tools import VectorArg, ScalarArg
        scan_knl_arguments = (
                [
                    # box-local morton bin counts for each particle at the current level
                    # only valid from scan -> postprocess
                    VectorArg(morton_bin_count_dtype, "morton_bin_counts"), # [nparticles]

                    # (local) morton nrs for each particle at the current level
                    # only valid from scan -> postprocess
                    VectorArg(np.uint8, "morton_nrs"), # [nparticles]

                    # segment flags
                    # invariant to sorting once set
                    # (particles are only reordered within a box)
                    VectorArg(np.uint8, "box_start_flags"), # [nparticles]

                    VectorArg(box_id_dtype, "box_ids"), # [nparticles]
                    VectorArg(box_id_dtype, "unsplit_box_ids"), # [nparticles]
                    VectorArg(box_id_dtype, "split_box_ids"), # [nparticles]

                    # per-box morton bin counts
                    VectorArg(morton_bin_count_dtype, "box_morton_bin_counts"), # [nparticles]

                    # particle# at which each box starts
                    VectorArg(particle_id_dtype, "box_starts"), # [nboxes]

                    # pointer to parent box
                    VectorArg(box_id_dtype, "parent_ids"), # [nboxes]

                    # number of particles in each box
                    VectorArg(particle_id_dtype,"box_particle_counts"), # [nboxes]

                    # number of boxes total
                    VectorArg(box_id_dtype, "box_count"), # [1]

                    ScalarArg(np.int32, "level"),
                    ScalarArg(particle_id_dtype, "max_particles_in_box"),
                    ScalarArg(bbox_dtype, "bbox"),
                    ]
                + [VectorArg(geo_dtype, ax) for ax in axis_names]
                )

        from pyopencl.scan import GenericScanKernel
        scan_kernel = GenericScanKernel(
                self.context, scan_dtype,
                arguments=scan_knl_arguments,
                input_expr="scan_t_from_particle(%s)"
                    % ", ".join([
                        "i", "level", "box_ids[i]", "*box_count",
                        "box_starts[box_ids[i]]",
                        "box_particle_counts[box_ids[i]]",
                        "max_particles_in_box",
                        "&bbox"
                        ]
                        +["%s[i]" % ax for ax in axis_names]),
                scan_expr="scan_t_add(a, b, across_seg_boundary)",
                neutral="scan_t_neutral()",
                is_segment_start_expr="box_start_flags[i]",
                output_statement=SCAN_OUTPUT_STMT_TPL.render(**codegen_args),
                preamble=scan_preamble, options=self.options)

        postproc_kernel_source = POSTPROC_KERNEL_TPL.render(**codegen_args)

        from pyopencl.elementwise import ElementwiseKernel
        postproc_kernel = ElementwiseKernel(
                self.context,
                scan_knl_arguments
                + [VectorArg(geo_dtype, "sorted_"+ax) for ax in axis_names],
                str(postproc_kernel_source), name="postproc",
                preamble=str(preamble), options=self.options)

        return _KernelInfo(
                scan_kernel=scan_kernel,
                morton_bin_count_dtype=morton_bin_count_dtype,
                postproc_kernel=postproc_kernel
                )

    # }}}

    # {{{ run control

    def __call__(self, queue, particles, max_particles_in_box):
        dimensions = len(particles)

        bbox = self.bbox_finder(particles).get()

        axis_names = AXIS_NAMES[:dimensions]

        # make bbox top end slightly larger, to ensure scaled
        # coordinates are alwyas < 1
        for ax in axis_names:
            extent = bbox["max_"+ax] - bbox["min_"+ax]
            bbox["max_"+ax] += 1e-4*extent

        # {{{ make kernel

        from pytools import single_valued
        geo_dtype = single_valued(coord.dtype for coord in particles)
        particle_id_dtype = np.uint32
        box_id_dtype = np.uint32
        knl_info = self.get_kernel_info(dimensions, geo_dtype, particle_id_dtype, box_id_dtype)

        # }}}

        nparticles = single_valued(len(coord) for coord in particles)

        morton_bin_counts = cl.array.empty(queue, nparticles, dtype=knl_info.morton_bin_count_dtype)
        morton_nrs = cl.array.empty(queue, nparticles, dtype=np.uint8)
        box_start_flags = cl.array.zeros(queue, nparticles, dtype=np.int8)
        box_ids = cl.array.zeros(queue, nparticles, dtype=np.uint32)
        unsplit_box_ids = cl.array.zeros(queue, nparticles, dtype=np.uint32)
        split_box_ids = cl.array.zeros(queue, nparticles, dtype=np.uint32)

        nboxes_guess = nparticles # a conservative guess

        box_morton_bin_counts = cl.array.empty(queue, nboxes_guess,
                dtype=knl_info.morton_bin_count_dtype)
        box_starts = cl.array.zeros(queue, nboxes_guess, dtype=np.uint32)
        parent_ids = cl.array.zeros(queue, nboxes_guess, dtype=np.uint32)
        box_particle_counts = cl.array.zeros(queue, nboxes_guess, dtype=np.uint32)
        cl.enqueue_copy(queue, box_particle_counts.data,
                box_particle_counts.dtype.type(nparticles))

        box_count = cl.array.empty(queue, 1, dtype=box_id_dtype)
        box_count.fill(1)

        from pytools.obj_array import make_obj_array

        box_count_last = None

        level = 0
        while True:
            args = ((morton_bin_counts, morton_nrs,
                    box_start_flags, box_ids, unsplit_box_ids, split_box_ids,
                    box_morton_bin_counts,
                    box_starts, parent_ids, box_particle_counts,
                    box_count,
                    level, max_particles_in_box, bbox)
                    + tuple(particles))
            knl_info.scan_kernel(*args)

            #print "split_box_ids", split_box_ids.get()[:nparticles]

            sorted_particles = make_obj_array([
                pt.copy() for pt in particles])
            knl_info.postproc_kernel(*(args + tuple(sorted_particles)))

            box_count_host = box_count.get()
            if 0:
                print "--------------LEVL"
                print "nboxes", box_count.get()
                print "box_ids", box_ids.get()[:nparticles]
                print "starts", box_starts.get()[:box_count_host]
                print "counts", box_particle_counts.get()[:box_count_host]

            if box_count_host == box_count_last:
                break

            particles = sorted_particles

            level += 1
            box_count_last = box_count_host
        print box_count_last

    # }}}

# }}}




# vim: filetype=pyopencl:fdm=marker
