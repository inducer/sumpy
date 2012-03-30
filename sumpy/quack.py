from __future__ import division

import numpy as np
import loopy as lp
import pyopencl as cl





def pop_expand(kernel, order, avec, bvec):
    dimensions = len(avec)
    from pytools import (
            generate_nonnegative_integer_tuples_summing_to_at_most
            as gnitstam)

    multi_indices = sorted(gnitstam(order, dimensions), key=sum)

    from sumpy.tools import mi_factorial, mi_power, mi_derivative
    return sum(
            mi_power(bvec, mi)/mi_factorial(mi) 
            * (-1)**sum(mi) # we're expanding K(-a)
            * mi_derivative(kernel, avec, mi)
            for mi in multi_indices)




class _KernelComputation:
    def __init__(self, kernel_getters, strength_usage,
            value_dtypes=np.float64, strength_dtypes=None):
        # {{{ process value_dtypes

        if not isinstance(value_dtypes, (list, tuple)):
            value_dtypes = [np.dtype(value_dtypes)] * len(kernel_getters)
        value_dtypes = [np.dtype(vd) for vd in value_dtypes]

        # }}}

        # {{{ process strength_usage

        if strength_usage is None:
            strength_usage = [0] * len(kernel_getters)

        if len(kernel_getters) != len(strength_usage):
            raise ValueError("exprs and strength_usage must have the same length")
        strength_count = max(strength_usage)+1

        # }}}

        # {{{ process strength_dtypes

        if strength_dtypes is None:
            strength_dtypes = value_dtypes[0]

        if not isinstance(strength_dtypes, (list, tuple)):
            strength_dtypes = [np.dtype(strength_dtypes)] * strength_count

        if len(strength_dtypes) != strength_count:
            raise ValueError("exprs and strength_usage must have the same length")

        # }}}

        self.kernel_getters = kernel_getters
        self.value_dtypes = value_dtypes
        self.strength_usage = strength_usage
        self.strength_dtypes = strength_dtypes




def get_direct_loopy_kernel(cl_device, dimensions,
        kernel_getters, strength_usage=None,
        geo_dtype=np.float64, value_dtypes=np.float64,
        strength_dtypes=None):
    """
    :arg kernel_getters: functions which return kernels as sympy expressions
      when given a :class:`sympy.Matrix`-type vector.
    :arg strength_usage: A list of integers indicating which expression
      uses which source strength indicator. This implicitly specifies the
      number of strength arrays that need to be passed.
      Default: all kernels use the same strength.
    """
    geo_dtype = np.dtype(geo_dtype)

    ki = _KernelComputation(kernel_getters, strength_usage,
            value_dtypes, strength_dtypes)

    from sumpy.symbolic import make_sym_vector

    avec = make_sym_vector("d", dimensions)

    kernels = [kg(avec) for kg in kernel_getters]
    from sumpy.codegen import (
            HANKEL_PREAMBLE, sympy_to_pymbolic_for_code)
    exprs = sympy_to_pymbolic_for_code([k for  k in kernels])
    from pymbolic import var
    exprs = [var("strength_%d" % i)[var("isrc")]*expr
            for i, expr in enumerate(exprs)]

    arguments = (
            [
               lp.ArrayArg("src", geo_dtype, shape=("nsrc", dimensions), order="C"),
               lp.ArrayArg("tgt", geo_dtype, shape=("ntgt", dimensions), order="C"),
               lp.ScalarArg("nsrc", np.int32),
               lp.ScalarArg("ntgt", np.int32),
               lp.ScalarArg("k", np.complex128),
            ]+[
               lp.ArrayArg("strength_%d" % i, dtype, shape="nsrc", order="C")
               for i, dtype in enumerate(ki.strength_dtypes)
            ]+[
               lp.ArrayArg("result_%d" % i, dtype, shape="ntgt", order="C")
               for i, dtype in enumerate(ki.value_dtypes)
               ])

    from pymbolic import parse
    knl = lp.make_kernel(cl_device,
            "[nsrc,ntgt] -> {[isrc,itgt,idim]: 0<=itgt<ntgt and 0<=isrc<nsrc "
            "and 0<=idim<%d}" % dimensions,
            [
            "[|idim] <%s> d[idim] = tgt[itgt,idim] - src[isrc,idim]" % geo_dtype.name,
            ]+[
            lp.Instruction(id=None,
                assignee=parse("pair_result_%d" % i), expression=expr,
                temp_var_type=dtype)
            for i, (expr, dtype) in enumerate(zip(exprs, ki.value_dtypes))
            ]+[
            "result_%d[itgt] = sum_%s(isrc, pair_result_%d)" % (i, dtype.name, i)
            for i, (expr, dtype) in enumerate(zip(exprs, ki.value_dtypes))],
            arguments,
           name="direct", assumptions="nsrc>=1 and ntgt>=1",
           preamble=["""
           #define PYOPENCL_DEFINE_CDOUBLE
           #include "pyopencl-complex.h"
           """, HANKEL_PREAMBLE])

    return knl




def get_pop_loopy_kernel(cl_device, dimensions,
        kernel_getters, order, strength_usage=None,
        geo_dtype=np.float64, value_dtypes=np.float64,
        strength_dtypes=None):
    """
    :arg kernel_getters: functions which return kernels as sympy expressions
      when given a :class:`sympy.Matrix`-type vector.
    :arg strength_usage: A list of integers indicating which expression
      uses which source strength indicator. This implicitly specifies the
      number of strength arrays that need to be passed.
      Default: all kernels use the same strength.
    """
    geo_dtype = np.dtype(geo_dtype)

    ki = _KernelComputation(kernel_getters, strength_usage,
            value_dtypes, strength_dtypes)

    from sumpy.symbolic import make_sym_vector

    avec = make_sym_vector("a", dimensions)
    bvec = make_sym_vector("b", dimensions)

    kernels = [kg(avec) for kg in kernel_getters]
    from sumpy.codegen import (
            HANKEL_PREAMBLE, sympy_to_pymbolic_for_code)
    exprs = sympy_to_pymbolic_for_code(
            [pop_expand(k, order, avec, bvec)
                for i, k in enumerate(kernels)])
    from pymbolic import var
    exprs = [var("strength_%d" % i)[var("isrc")]*expr
            for i, expr in enumerate(exprs)]

    arguments = (
            [
               lp.ArrayArg("src", geo_dtype, shape=("nsrc", dimensions), order="C"),
               lp.ArrayArg("tgt", geo_dtype, shape=("ntgt", dimensions), order="C"),
               lp.ArrayArg("center", geo_dtype, shape=("ntgt", dimensions), order="C"),
               lp.ScalarArg("nsrc", np.int32),
               lp.ScalarArg("ntgt", np.int32),
               lp.ScalarArg("k", np.complex128),
            ]+[
               lp.ArrayArg("strength_%d" % i, dtype, shape="nsrc", order="C")
               for i, dtype in enumerate(ki.strength_dtypes)
            ]+[
               lp.ArrayArg("result_%d" % i, dtype, shape="ntgt", order="C")
               for i, dtype in enumerate(ki.value_dtypes)
               ])

    from pymbolic import parse
    knl = lp.make_kernel(cl_device,
            "[nsrc,ntgt] -> {[isrc,itgt,idim]: 0<=itgt<ntgt and 0<=isrc<nsrc "
            "and 0<=idim<%d}" % dimensions,
            [
            "[|idim] <%s> a[idim] = center[itgt,idim] - src[isrc,idim]" % geo_dtype.name,
            "[|idim] <%s> b[idim] = tgt[itgt,idim] - center[itgt,idim]" % geo_dtype.name,
            ]+[
            lp.Instruction(id=None,
                assignee=parse("pair_result_%d" % i), expression=expr,
                temp_var_type=dtype)
            for i, (expr, dtype) in enumerate(zip(exprs, ki.value_dtypes))
            ]+[
            "result_%d[itgt] = sum_%s(isrc, pair_result_%d)" % (i, dtype.name, i)
            for i, (expr, dtype) in enumerate(zip(exprs, ki.value_dtypes))],
            arguments,
           name="pop", assumptions="nsrc>=1 and ntgt>=1",
           preamble=["""
           #define PYOPENCL_DEFINE_CDOUBLE
           #include "pyopencl-complex.h"
           """, HANKEL_PREAMBLE])

    return knl




def test_pop_kernel(ctx_factory):
    ctx = ctx_factory()
    dimensions = 2

    from sumpy.symbolic import make_helmholtz_kernel

    knl = get_pop_loopy_kernel(ctx.devices[0], dimensions,
            [make_helmholtz_kernel], 5, value_dtypes=np.complex128)

    ref_knl = knl

    def variant_1(knl):
        knl = lp.split_dimension(knl, "itgt", 1024, outer_tag="g.0")
        return knl

    def variant_2(knl):
        knl = lp.split_dimension(knl, "itgt", 256,
                outer_tag="g.0", inner_tag="l.0", slabs=(0,1))
        knl = lp.split_dimension(knl, "isrc", 256, slabs=(0,1))
        knl = lp.add_prefetch(knl, "tgt[itgt,k]", ["k"], default_tag=None)
        #knl = lp.add_prefetch(knl, "x[j,k]", ["j_inner", "k"],
                #["x_fetch_j", "x_fetch_k"])

        return knl

    knl = variant_1(knl)

    nsrc = 3000
    ntgt = 3000

    fake_par_values = dict(nsrc=nsrc, ntgt=ntgt, k=1)

    kernel_gen = lp.generate_loop_schedules(knl)
    kernel_gen = lp.check_kernels(kernel_gen, fake_par_values)

    lp.auto_test_vs_ref(ref_knl, ctx, kernel_gen,
            op_count=[nsrc*ntgt], op_label=["point pairs"],
            parameters=fake_par_values, print_ref_code=True,
            codegen_kwargs=dict(allow_complex=True))




def test_direct(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    dimensions = 2

    from sumpy.symbolic import make_laplace_kernel

    knl = get_direct_loopy_kernel(ctx.devices[0], dimensions,
            [make_laplace_kernel], value_dtypes=np.complex128)
    knl = lp.split_dimension(knl, "itgt", 1024, outer_tag="g.0")

    cknl = lp.CompiledKernel(ctx, knl)

    src = np.random.randn(300, 2)*3
    src[1] = 1
    charge = np.ones((len(src),), dtype=np.complex128)

    center = np.asarray([0,0], dtype=np.float64)
    from hellskitchen.visualization import FieldPlotter
    fp = FieldPlotter(center, points=1000, extent=5)

    tgt = fp.points.copy()

    evt, pot = cknl(queue, src=src, tgt=tgt, nsrc=len(src), ntgt=len(tgt), k=17,
            strength_0=charge, out_host=True)

    plotval = np.log(1e-15+np.abs(pot))
    fp.show_scalar_in_matplotlib(plotval.real)
    import matplotlib.pyplot as pt
    pt.show()




if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
