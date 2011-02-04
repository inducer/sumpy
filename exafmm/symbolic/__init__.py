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
        return type(expr)(self.rec(expr.p), self.rec(expr.q))

    def map_Integer(self, expr):
        return expr
    map_int = map_Integer

    map_Symbol = map_Integer
    map_Real = map_Integer




def make_coulomb_kernel(dimensions=3):
    from exafmm.symbolic import make_sym_vector
    tgt = make_sym_vector("t", dimensions)
    src = make_sym_vector("s", dimensions)

    return 1/sp.sqrt(((tgt-src).T*(tgt-src))[0,0])





