from __future__ import division
from pytools import memoize




def make_sym_vector(name, components):
    return sp.Matrix(
            [sp.Symbol("%s%d" % (name, i)) for i in range(components)])




class SympyMapper(object):
    def __call__(self, expr, *args, **kwargs):
        return self.rec(expr, *args, **kwargs)

    def rec(self, expr, *args, **kwargs):
        mro = list(type(expr).__mro__)
        dispatch_class = kwargs.pop("dispatch_class", type(self))

        while mro:
            method_name = "map_"+mro.pop(0).__name__

            try:
                method = getattr(dispatch_class, method_name)
            except AttributeError:
                pass
            else:
                return method(self, expr, *args, **kwargs)

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
    map_ImaginaryUnit = map_Integer



def vector_subs(expr, old, new):
    from pymbolic.mapper.substitutor import substitute
    return substitute(expr, dict(
        (old_i, new_i)
        for old_i, new_i in zip(old, new)))




def find_power_of(base, prod):
    remdr = sp.Wild("remdr")
    power = sp.Wild("power")
    result = prod.match(remdr*base**power)
    if result is None:
        return 0
    return result[power]
