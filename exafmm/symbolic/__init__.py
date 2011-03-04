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




def vector_subs(expr, old, new):
    result = expr
    for old_i, new_i in zip(old, new):
        result = result.subs(old_i, new_i)

    return result




class CSEMapper(IdentityMapper):
    def __init__(self, to_eliminate, sym_gen):
        self.to_eliminate = to_eliminate
        self.sym_gen = sym_gen

        self.cse_assignments = []
        self.subexpr_symbols = {}

    def get_cse(self, expr):
        try:
            return self.subexpr_symbols[expr]
        except KeyError:
            new_expr = self.rec(expr, dispatch_class=IdentityMapper)
            new_sym = self.sym_gen.next()
            self.cse_assignments.append(
                (new_sym.name, expr))
            self.subexpr_symbols[expr] = new_sym
            return new_sym

    def map_Add(self, expr):
        if expr in self.to_eliminate:
            return self.get_cse(expr)
        else:
            return IdentityMapper.map_Add(self, expr)

    map_Mul = map_Add
    map_Pow = map_Add
    map_Function = map_Add

    def map_Rational(self, expr):
        if expr in self.to_eliminate:
            return self.get_cse(expr)
        else:
            return IdentityMapper.map_Rational(self, expr)




def eliminate_common_subexpressions(exprs, sym_gen):
    from sympy.utilities.iterables import postorder_traversal

    subexpr_count = {}
    for expr in exprs:
        for subexpr in postorder_traversal(expr):
            if not subexpr.is_Symbol:
                subexpr_count[subexpr] = \
                    subexpr_count.get(subexpr, 0) + 1

    cse_mapper = CSEMapper(
        set([subexpr 
        for subexpr, count in subexpr_count.iteritems()
        if count > 1]),
        sym_gen)
    result = [cse_mapper(expr) for expr in exprs]
    return cse_mapper.cse_assignments, result




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

