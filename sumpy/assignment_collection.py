from __future__ import division

import sympy as sp

from pytools import Record
from pymbolic import var
from sumpy.symbolic import SympyIdentityMapper





def get_tuples_below_or_equal(tp):
    if not tp:
        yield ()
        return

    first_len = tp[0]
    rest = tp[1:]
    for subtp in get_tuples_below_or_equal(rest):
        for i in xrange(first_len+1):
            yield (i,)+subtp




class _DependencyToFunctionCallMapper(SympyIdentityMapper):
    """The only viable way to let sympy know that some variable depends on
    another with a known partial derivative is to turn it into a function call,
    with arguments consisting of the dependencies. This mapper performs that
    function.
    """

    def __init__(self, get_all_deps):
        self.get_all_deps = get_all_deps

    def map_Symbol(self, expr):
        deps = list(self.get_all_deps(expr.name))
        if not deps:
            return expr
        else:
            deps = list(deps)
            return sp.Function(expr.name)(*deps)




class _DerivativePostProcessor(SympyIdentityMapper):
    def __init__(self, sac):
        self.sac = sac

    def map_Function(self, expr):
        name = type(expr).__name__
        if name in self.sac.assignments:
            return sp.Symbol(name)
        else:
            return SympyIdentityMapper.map_Function(self, expr)

    def generate_standard_dummies(self):
        i = 1
        while True:
            yield sp.Symbol("xi_%d" % i)
            i += 1

    def map_Derivative(self, expr):
        assert isinstance(expr.expr, sp.Function)

        func_name = type(expr.expr).__name__

        wrt = {}
        for wrt_var in expr.variables:
            wrt_var = self.rec(wrt_var)
            assert isinstance(wrt_var, sp.Symbol)
            wrt[wrt_var.name] = wrt.get(wrt_var.name, 0) + 1

        if func_name in self.sac.assignments:
            return self.sac.get_derivative(func_name, wrt.iteritems())
        else:
            substs = {}
            point = []
            gen_dummy = iter(self.generate_standard_dummies())

            new_point_vars = []
            for wrt in expr.variables:
                if isinstance(wrt, sp.Function):
                    if wrt in substs:
                        new_dummy = substs[wrt]
                    else:
                        new_dummy = gen_dummy.next()
                        substs[wrt] = new_dummy
                        new_point_vars.append(new_dummy)
                        point.append(self.rec(wrt))
                else:
                    new_point_vars.append(wrt)
                    point.append(self.rec(wrt))

            return sp.Subs(
                    expr.subs(substs.items()),
                    new_point_vars, point)

    def map_Subs(self, expr):
        # Subs'd derivatives are none of our beeswax--they are external
        # functions, not fake-functions inserted by
        # _DependencyToFunctionCallMapper. Assert this fact.

        assert isinstance(expr.expr, sp.Derivative)
        assert isinstance(expr.expr.expr, sp.Function)

        func = expr.expr.expr
        assert type(func).__name__ not in self.sac.assignments

        return sp.Subs(expr.expr, expr.variables,
                tuple(self.rec(pt_i) for pt_i in expr.point))





def _generate_unique_possibilities(prefix):
    try_num = 0
    while True:
        yield "%s%d" % (prefix, try_num)
        try_num += 1




class _SymbolGenerator:
    def __init__(self, taken_symbols):
        self.taken_symbols = taken_symbols
        self.generated_names = set()

    def __call__(self, base="expr"):
        for id_str in _generate_unique_possibilities(base):
            if id_str not in self.taken_symbols and id_str not in self.generated_names:
                self.generated_names.add(id_str)
                yield sp.Symbol(id_str)




class SymbolicAssignmentCollection(Record):
    """Represents a collection of assignments::

        a = 5*x
        b = a**2-k

    In the above, *x* and *k* are external variables, and *a* and *b*
    are variables managed by this object. In addition to managing
    such a collection, this class also provides a facility for computing
    (in a somewhat efficient manner) high derivatives of managed with
    respect to unmanaged variables.

    This is a stateful object, but the only state changes allowed
    are additions to *assignments*, and corresponding updates of
    *partial_to_name*.

    Note that user code is *only* allowed to hold on to *names* generated
    by this class, but not expressions using names defined in this collection.
    """

    def __init__(self, assignments=None, partial_to_name=None):
        """
        :arg assignments: mapping from *var_name* to expression
        :arg partial_to_name: mapping from *(var_name, wrt)*
            to variable name containing the derivative of *expr*
            with respect to *wrt*, where *wrt* is a :class:`frozenset`
            of *(var_name, count)* tuples
        """

        if assignments is None:
            assignments = {}

        if partial_to_name is None:
            partial_to_name = dict(
                    ((name, frozenset()), name)
                    for name in assignments)

        from pytools import reverse_dictionary
        Record.__init__(self,
                assignments=assignments,
                partial_to_name=partial_to_name,
                name_to_partial=reverse_dictionary(partial_to_name))

        self.nesting_level = 0
        self.symbol_generator = _SymbolGenerator(self.assignments)()
        self.all_dependencies_cache = {}

    def get_all_dependencies(self, var_name):
        """Including recursive dependencies."""
        try:
            return self.all_dependencies_cache[var_name]
        except KeyError:
            pass

        if var_name not in self.assignments:
            return set()

        result = set()
        for dep in self.assignments[var_name].atoms():
            if not isinstance(dep, sp.Symbol):
                continue

            dep_name = dep.name
            if dep_name in self.assignments:
                result.update(self.get_all_dependencies(dep_name))
            else:
                result.add(dep)

        self.all_dependencies_cache[var_name] = result
        return result

    def add_assignment(self, name, expr, root_name=None, wrt_set=None):
        assert isinstance(name, str)
        assert name not in self.assignments
        assert not isinstance(expr, sp.Symbol)
        assert not expr.is_Number

        if wrt_set is None:
            wrt_set = frozenset()
        if root_name is None:
            root_name = name

        #print "NEW_ASSIGN", name
        #sp.pretty_print(expr)

        self.assignments[name] = expr
        self.partial_to_name[root_name, wrt_set] = name
        self.name_to_partial[name] = (root_name, wrt_set)

    def assign_unique(self, name_base, expr):
        """Assign *expr* to a new variable whose name is based on *name_base*.
        Return the new variable name.
        """
        for new_name in _generate_unique_possibilities(name_base):
            if new_name not in self.assignments:
                break

        self.add_assignment(new_name, expr)
        return new_name

    def get_derivative(self, var_name, wrt):
        self.nesting_level += 1
        result = self._get_derivative(var_name, wrt)
        self.nesting_level -= 1
        return result

    def _get_derivative(self, var_name, wrt):
        """
        :arg wrt: an iterable of *(var_name, count)* tuples
        :returns: variable name of src derivative

        Note: only derivatives with respect to external (non-assigned)
        variables are supported.
        """

        prefix = "|  "* self.nesting_level
        prefix = prefix[:-1]

        # {{{ normalize var_name and wrt to be relative to a root expression

        root_name, root_wrt = self.name_to_partial[var_name]
        root_wrt = dict(root_wrt)
        for wrt_name, wrt_count in wrt:
            assert wrt_name not in self.assignments
            root_wrt[wrt_name] = root_wrt.get(wrt_name, 0) + wrt_count
        root_wrt_frozenset = frozenset(
                (wrt_name, count) for wrt_name, count in root_wrt.iteritems()
                if count)

        # }}}

        #print prefix, "ENTER", var_name, root_name, root_wrt

        # {{{ early exit if derivative is already known

        try:
            result = self.partial_to_name[root_name, root_wrt_frozenset]
        except KeyError:
            pass
        else:
            if isinstance(result, sp.Basic):
                assert result.is_Number
                return result

            #print prefix, "KNOWN ->", result
            return sp.Symbol(result)

        # }}}

        # {{{ find shortest "path" to known derivative of root

        list_root_wrt = list(root_wrt.iteritems())
        root_wrt_names = list(vn for (vn, count) in list_root_wrt)
        root_wrt_counts = list(count for (vn, count) in list_root_wrt)

        # shorter paths first
        paths = sorted(get_tuples_below_or_equal(root_wrt_counts), key=sum)

        found = False
        for path in paths:
            src_wrt = frozenset((name, rwc-i)
                    for name, rwc, i in zip(root_wrt_names, root_wrt_counts, path)
                    if rwc-i)

            if (root_name, src_wrt) in self.partial_to_name:
                found = True
                break

        assert found

        # If the path length to a known derivative is 0, that means we already
        # know that derivative. Return it.

        # }}}

        # Find a nonzero entry in 'path', i.e. a derivative
        # we are going to take right here.
        for diff_name, diff_cnt in zip(root_wrt_names, path):
            if diff_cnt:
                break

        base_root_wrt = root_wrt.copy()
        base_root_wrt[diff_name] -= 1

        base_symbol = self.get_derivative(root_name, frozenset(base_root_wrt.iteritems()))
        assert isinstance(base_symbol, sp.Symbol)
        base_name = base_symbol.name

        base_expr = self.assignments[base_name]

        dep_adder = _DependencyToFunctionCallMapper(self.get_all_dependencies)
        with_dep = dep_adder(base_expr)
        result = with_dep.diff(sp.Symbol(diff_name))

        #print prefix, "BEFORE POSTPROC"
        #sp.pretty_print(result)

        postproc = _DerivativePostProcessor(self)
        result = postproc(result)

        #print prefix, "AFTER POSTPROC"
        #sp.pretty_print(result)

        # Substitute in already known variables
        result = result.subs(
                [(expr, name) for name, expr in self.assignments.iteritems()])

        result, = self.run_global_cse([result])

        if isinstance(result, sp.Symbol):
            self.partial_to_name[root_name, root_wrt_frozenset] = result.name
            return result
        if result.is_Number:
            self.partial_to_name[root_name, root_wrt_frozenset] = result
            return result

        new_name = "d%s_%s" % (
                root_name, "_".join("d%d%s" % (count, name)
                    for name, count in list_root_wrt))

        self.add_assignment(new_name, result, root_name, root_wrt_frozenset)

        #raw_input("ENTER: ")

        #print prefix, "LEAVE"
        return sp.Symbol(new_name)

    def run_global_cse(self, extra_exprs):
        assign_names = list(self.assignments)
        assign_exprs = [self.assignments[name] for name in assign_names]
        new_assignments, new_exprs = sp.cse(assign_exprs + extra_exprs, symbols=self.symbol_generator)

        # {{{ check if CSEs are complex enough to warrant being stored

        approved_assignments = []
        rejected_assignments = []
        for name, value in new_assignments:
            approved = True

            # const*var: not good enough
            if (isinstance(value, sp.Mul)
                    and len(value.args) == 2
                    and sum(1 for arg in value.args if arg.is_Number) == 1
                    and sum(1 for arg in value.args if isinstance(arg, sp.Symbol)) == 1
                    ):
                approved = False

            if approved:
                approved_assignments.append((name, value))
            else:
                rejected_assignments.append((name, value))

        # un-substitute rejected assignments
        unsubst_rej = reversed(rejected_assignments)
        new_exprs = [expr.subs(unsubst_rej) for expr in new_exprs]
        approved_assignments = [(name, expr.subs(unsubst_rej))
                for name, expr in approved_assignments]

        # }}}

        new_assign_exprs = new_exprs[:len(assign_exprs)]
        new_extra_exprs = new_exprs[len(assign_exprs):]

        for name, old_expr, new_expr in zip(assign_names, assign_exprs, new_assign_exprs):
           self.assignments[name] = new_expr

        for name, value in approved_assignments:
            assert isinstance(name, sp.Symbol)
            self.add_assignment(name.name, value)

        return new_extra_exprs



# vim: fdm=marker
