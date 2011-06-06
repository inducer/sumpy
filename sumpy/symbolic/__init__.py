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




COMMUTATIVE_CLASSES = (sp.Add, sp.Mul)




def get_normalized_cse_key(node):
    if isinstance(node, COMMUTATIVE_CLASSES):
        return type(node), frozenset(node.args)
    else:
        return node




class CSEMapper(IdentityMapper):
    def __init__(self, to_eliminate, sym_gen):
        self.to_eliminate = to_eliminate
        self.sym_gen = sym_gen

        self.cse_assignments = []
        self.subexpr_symbols = {}

    def get_cse(self, expr, key=None):
        if key is None:
            key = get_normalized_cse_key(expr)

        try:
            return self.subexpr_symbols[key]
        except KeyError:
            new_expr = self.rec(expr, dispatch_class=IdentityMapper)
            new_sym = self.sym_gen.next()
            self.cse_assignments.append(
                (new_sym.name, new_expr))
            self.subexpr_symbols[key] = new_sym
            return new_sym

    def map_Add(self, expr):
        key = get_normalized_cse_key(expr)
        if key in self.to_eliminate:
            result = self.get_cse(expr, key)
            return result
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
    subexpr_count = {}

    def gather_use_counts(node):
        """Like :func:`sympy.utilities.iterables.postorder_traversal`,
        but returns tuples of (level, node) which indicate the level
        of nesting.
        """

        key = get_normalized_cse_key(node)

        if key in subexpr_count:
            subexpr_count[key] += 1
            # do not re-traverse (and thus re-count subexpressions)
        else:
            subexpr_count[key] = 1

            if isinstance(node, sp.Basic):
                iterable = node.args
            else:
                iterable = node

            for arg in node.args:
                gather_use_counts(arg)

    for expr in exprs:
        gather_use_counts(expr)

    to_eliminate = set([subexpr_key
        for subexpr_key, count in subexpr_count.iteritems()
        if count > 1])
    cse_mapper = CSEMapper(to_eliminate, sym_gen)
    result = [cse_mapper(expr) for expr in exprs]
    return cse_mapper.cse_assignments, result




def make_coulomb_kernel_in(var_name, dimensions):
    from sumpy.symbolic import make_sym_vector
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

