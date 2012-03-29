from __future__ import division
import sympy as sp
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

    def map_Subs(self, expr):
        return expr




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

            for arg in iterable:
                gather_use_counts(arg)

    for expr in exprs:
        gather_use_counts(expr)

    to_eliminate = set([subexpr_key
        for subexpr_key, count in subexpr_count.iteritems()
        if count > 1])
    cse_mapper = CSEMapper(to_eliminate, sym_gen)
    result = [cse_mapper(expr) for expr in exprs]
    return cse_mapper.cse_assignments, result




def find_power_of(base, prod):
    remdr = sp.Wild("remdr")
    power = sp.Wild("power")
    result = prod.match(remdr*base**power)
    if result is None:
        return 0
    return result[power]




@memoize
def diff_multi_index(expr, multi_index, var_name):
    dimensions = len(multi_index)

    if sum(multi_index) == 0:
        return expr

    first_nonzero_axis = min(
            i for i in range(dimensions)
            if multi_index[i] > 0)

    lowered_mi = list(multi_index)
    lowered_mi[first_nonzero_axis] -= 1
    lowered_mi = tuple(lowered_mi)

    lower_diff_expr = diff_multi_index(expr, lowered_mi, var_name)

    return sp.diff(lower_diff_expr,
            sp.Symbol("%s%d" % (var_name, first_nonzero_axis)))




def make_laplace_kernel(dist_vec):
    dimensions = len(dist_vec)
    r = sp.sqrt((dist_vec.T*dist_vec)[0,0])

    if dimensions == 2:
        return sp.log(r)
    elif dimensions == 3:
        return 1/r
    else:
        raise RuntimeError("unsupported dimensionality")





def make_helmholtz_kernel(dist_vec):
    dimensions = len(dist_vec)
    r = sp.sqrt((dist_vec.T*dist_vec)[0,0])

    i = sp.sqrt(-1)
    k = sp.Symbol("k")

    if dimensions == 2:
        return i/4 * sp.Function("H1_0")(k*r)
    elif dimensions == 3:
        return sp.exp(i*k*r)/r
    else:
        raise RuntimeError("unsupported dimensionality")




def make_coulomb_kernel_in(var_name, dimensions):
    from warnings import warn
    warn("make_coulomb_kernel_in is deprecated", DeprecationWarning, stacklevel=2)

    from sumpy.symbolic import make_sym_vector
    return make_laplace_kernel(make_sym_vector(var_name, dimensions))




def make_helmholtz_kernel_in(var_name, dimensions):
    from warnings import warn
    warn("make_helmholtz_kernel_in is deprecated", DeprecationWarning, stacklevel=2)

    from sumpy.symbolic import make_sym_vector
    return make_helmholtz_kernel(make_sym_vector(var_name, dimensions))




def make_coulomb_kernel_ts(dimensions):
    old = make_sym_vector("d", dimensions)
    new = (make_sym_vector("t", dimensions)
            - make_sym_vector("s", dimensions))

    return vector_subs(make_coulomb_kernel_in("d", dimensions),
            old, new)

