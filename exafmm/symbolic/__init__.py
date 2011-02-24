from __future__ import division
import sympy as sp




def make_sym_vector(name, components):
    return sp.Matrix(
            [sp.Symbol("%s%d" % (name, i)) for i in range(components)])





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

    def map_Rational(self, expr):

        # Can't use type(expr) here because sympy has a class 'Half'
        # that embodies just the rational number 1/2.

        return sp.Rational(self.rec(expr.p), self.rec(expr.q))

    def map_Integer(self, expr):
        return expr
    map_int = map_Integer

    map_Symbol = map_Integer
    map_Real = map_Integer




def vector_subs(expr, old, new):
    result = expr
    for old_i, new_i in zip(old, new):
        result = result.subs(old_i, new_i)

    return result





# Kernel symbolic variable name conventions:
#
# si: sources
# ti: targets
# ci: center of the expansion (xc in Rio's notes)
#
# di = ti - si
# ai = ci - si    (x-X in Rio's notes)
# bi = ti - ci    (X in Rio's notes)
#
# a takes you from source to center of expansion
# b takes you from center of expansion to target
# 
# (i is an axis number, 0 for x, 1 for y, etc.)




def make_coulomb_kernel_in(var_name, dimensions):
    from exafmm.symbolic import make_sym_vector
    dist = make_sym_vector(var_name, dimensions)

    if dimensions == 2:
        return sp.log(sp.sqrt((dist.T*dist)[0,0]))
    else:
        return 1/sp.sqrt((dist.T*dist)[0,0])




def make_coulomb_kernel_ts(dimensions):
    old = make_sym_vector("d", dimensions)
    new = (make_sym_vector("t", dimensions)
            - make_sym_vector("s", dimensions))

    return vector_subs(make_coulomb_kernel_in("d", dimensions),
            old, new)

