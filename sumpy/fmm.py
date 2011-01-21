from __future__ import division

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import sympy as sp
import sympy.printing.ccode
import numpy.linalg as la

# TODO:
# - Data layout, float4s bad
# - Make side-effect-free
# - Exclude self-interaction if source and target are same

# LATER:
# - Optimization for source = target (postpone)


DIRECT_KERNEL = """
    __kernel void sum_direct(
      __global float *potential_g,
      __global const float4 *target_g,
      __global const float4 *source_g,
      ulong nsource,
      ulong ntarget)
    {
      int itarget = get_global_id(0);
      if (itarget >= ntarget) return;

      float p=0;
      for(int isource=0; isource<nsource; isource++ )
      {
        float4 dist = target_g[itarget] - source_g[isource];
        float4 dist_sq = dist*dist;
        p += source_g[isource].w * rsqrt(dist_sq.x + dist_sq.y + dist_sq.z);
      }
      potential_g[itarget] = p;
    }
    """



def make_sym_vector(name, components):
    return sp.Matrix(
            [sp.Symbol("%s%d" % (name, i)) for i in range(components)])



class CLCodePrinter(sp.printing.codeprinter.CodePrinter):
    def __init__(self, vectors=set()):
        sp.printing.codeprinter.CodePrinter.__init__(self)
        self.vectors = vectors
 
    def _print_Pow(self, expr):
        from sympy.core import S
        from sympy.printing.precedence import precedence
        PREC = precedence(expr)
        if expr.exp is S.NegativeOne:
            return '1.0/%s'%(self.parenthesize(expr.base, PREC))
        elif expr.exp == 0.5:
            return 'sqrt(%s)' % self._print(expr.base)
        elif expr.exp == -0.5:
            return 'rsqrt(%s)' % self._print(expr.base)
        elif expr.exp == 2 and isinstance(expr.base, sp.Symbol):
            return '%s*%s' % (expr.base.name, expr.base.name)
        else:
            return 'pow(%s, %s)'%(self._print(expr.base),
                                 self._print(expr.exp))




class SympyMapper(object):
    def __call__(self, expr, *args, **kwargs):
        return self.rec(expr, *args, **kwargs)

    def rec(self, expr, *args, **kwargs):
        mro = list(type(expr).__mro__)

        while mro:
            method_name = "map_"+mro.pop(0).__name__

            try:
                method = getattr(self, method_name)
            except AttributeError:
                pass
            else:
                return method(expr, *args, **kwargs)

        raise NotImplementedError(
                "%s does not know how to map type '%s'"
                % (type(self).__name__,
                    type(expr).__name__))





class IdentityMapper(SympyMapper):
    def map_Add(self, expr):
        return type(expr)(*tuple(self.rec(arg) for arg in expr.args))

    map_Mul = map_Add
    map_Pow = map_Add
    map_Function = map_Add

    def map_Integer(self, expr):
        return expr

    map_Symbol = map_Integer
    map_Real = map_Integer




class SquareRewriter(IdentityMapper):
    def __init__(self, symbol_gen, expr_to_var={}):
        self.assignments = []
        self.symbol_gen = iter(symbol_gen)
        self.expr_to_var = expr_to_var

    def get_var_for(self, expr):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            sym = self.symbol_gen.next()
            self.assignments.append((sym, expr))
            self.expr_to_var[expr] = sym
            return sym

    def __call__(self, var_name, expr):
        self.assignments.append((var_name, self.rec(expr)))

    def map_Pow(self, expr):
        if expr.exp == 2:
            new_base = self.get_var_for(expr.base)
            return new_base**2
        else:
            return IdentityMapper.map_Pow(self, expr)





def generate_cl_statements_from_assignments(assignments):
    """
    :param assignments: a list of tuples *(var_name, expr)*
    """

    # {{{ perform CSE

    from sympy.utilities.iterables import  numbered_symbols
    sym_gen = numbered_symbols("cse")

    new_assignments = []
    for var_name, expr in assignments:
        print 'Initial expression for',var_name
        print expr
        from sympy.simplify.cse_main import cse
        replacements, reduced = cse([expr], sym_gen)
        print 'replacements', replacements
        print 'reduced', reduced
        new_assignments.extend(
                (sym.name, expr) for sym, expr in replacements)
        print 'new_assignments', new_assignments
        new_assignments.append((var_name, reduced[0]))
        print new_assignments

    assignments = new_assignments
    print 'After CSE',assignments
    # }}}

    # {{{ rewrite squares

    sq_rewriter = SquareRewriter(sym_gen, expr_to_var=dict(
        (expr, sp.Symbol(var_name)) for var_name, expr in assignments))

    for var_name, expr in assignments:
        sq_rewriter(var_name, expr)
    print 'After SqRewrite',sq_rewriter.assignments

    # }}}

    # {{{ print code

    ccp = CLCodePrinter()
    return ["%s = %s" % (var_name, ccp.doprint(expr))
            for var_name, expr in sq_rewriter.assignments]

    # }}}

def gen_direct_sum_for_kernel(expr):
    lines = generate_cl_statements_from_assignments(
            [("result", expr)])

    print "\n".join(lines)

    1/0



def make_coulomb_kernel(dimensions=3):
    tgt = make_sym_vector("t", dimensions)
    src = make_sym_vector("s", dimensions)

    return 1/sp.sqrt(((tgt-src).T*(tgt-src))[0,0])





def test_direct():
    target = np.random.rand(5000, 4).astype(np.float32)
    source = np.random.rand(5000, 4).astype(np.float32)

    dev = cl.get_platforms()[0].get_devices()[1]
    print dev.name
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)

    target_dev = cl_array.to_device(ctx, queue, target)
    source_dev = cl_array.to_device(ctx, queue, source)



    prg = cl.Program(ctx,
            gen_direct_sum_for_kernel(
                    make_coulomb_kernel()
                    .diff(sp.Symbol("t0"))
                    )).build()

    sum_direct = prg.sum_direct
    sum_direct.set_scalar_arg_dtypes([None, None, None, np.uintp, np.uintp])

    potential_dev = cl_array.empty(ctx, len(target), np.float32, queue=queue)
    grp_size = 128
    sum_direct(queue, ((len(target) + grp_size) // grp_size * grp_size,), (grp_size,),
        potential_dev.data, target_dev.data, source_dev.data, len(source), len(target))

    potential = potential_dev.get()
    potential_host = np.empty_like(potential)

    for itarg in xrange(len(target)):
        potential_host[itarg] = np.sum(
                source[:,3]
                /
                np.sum((target[itarg,:3] - source[:,:3])**2, axis=-1)**0.5)

    #print potential[:100]
    #print potential_host[:100]
    assert la.norm(potential - potential_host)/la.norm(potential_host) < 1e-6

def test_symbolic():
    from sympy.utilities.iterables import numbered_symbols
    dim         = 3
    potKernel   = make_coulomb_kernel()
    fieldKernel = sp.Matrix([potKernel.diff(s) for d,s in zip(range(dim), numbered_symbols('t'))])
    print 'Kernels:'
    print potKernel
    print fieldKernel
    print 'Kernel OpenCL code:'
    lines = generate_cl_statements_from_assignments([('potential', potKernel)])
    print '\n'.join(lines)
    return

if __name__ == "__main__":
    test_symbolic()
    #test_direct()

# vim: foldmethod=marker
