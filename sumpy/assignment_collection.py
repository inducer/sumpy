from __future__ import division

import sympy as sp
from sumpy.symbolic import SympyIdentityMapper





def _generate_unique_possibilities(prefix):
    yield prefix

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




class SymbolicAssignmentCollection(object):
    """Represents a collection of assignments::

        a = 5*x
        b = a**2-k

    In the above, *x* and *k* are external variables, and *a* and *b*
    are variables managed by this object.

    This is a stateful object, but the only state changes allowed
    are additions to *assignments*, and corresponding updates of
    its lookup tables.

    Note that user code is *only* allowed to hold on to *names* generated
    by this class, but not expressions using names defined in this collection.
    """

    def __init__(self, assignments=None):
        """
        :arg assignments: mapping from *var_name* to expression
        """

        if assignments is None:
            assignments = {}

        self.assignments = assignments

        self.symbol_generator = _SymbolGenerator(self.assignments)()
        self.all_dependencies_cache = {}

        self.user_symbols = set()

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

        if wrt_set is None:
            wrt_set = frozenset()
        if root_name is None:
            root_name = name

        self.assignments[name] = expr

    def assign_unique(self, name_base, expr):
        """Assign *expr* to a new variable whose name is based on *name_base*.
        Return the new variable name.
        """
        for new_name in _generate_unique_possibilities(name_base):
            if new_name not in self.assignments:
                break

        self.add_assignment(new_name, expr)
        self.user_symbols.add(new_name)
        return new_name

    def run_global_cse(self, extra_exprs=[]):
        assign_names = list(self.assignments)
        assign_exprs = [self.assignments[name] for name in assign_names]
        new_assignments, new_exprs = sp.cse(assign_exprs + extra_exprs,
                symbols=self.symbol_generator)

        new_assign_exprs = new_exprs[:len(assign_exprs)]
        new_extra_exprs = new_exprs[len(assign_exprs):]

        for name, new_expr in zip(assign_names, new_assign_exprs):
           self.assignments[name] = new_expr

        for name, value in new_assignments:
            assert isinstance(name, sp.Symbol)
            self.add_assignment(name.name, value)

        return new_extra_exprs

    def kill_trivial_assignments(self, exprs):
        approved_assignments = []
        rejected_assignments = []

        from sumpy.symbolic import is_assignment_nontrivial
        for name, value in self.assignments.iteritems():
            if name in self.user_symbols or is_assignment_nontrivial(name, value):
                approved_assignments.append((name, value))
            else:
                rejected_assignments.append((name, value))

        # un-substitute rejected assignments
        from sumpy.symbolic import make_one_step_subst
        unsubst_rej = make_one_step_subst(rejected_assignments)

        new_assignments = dict(
                (name, expr.subs(unsubst_rej))
                for name, expr in approved_assignments)

        exprs = [expr.subs(unsubst_rej) for expr in exprs]
        self.assignments = new_assignments
        return exprs





# vim: fdm=marker
