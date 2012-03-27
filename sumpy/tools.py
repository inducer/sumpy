from __future__ import division




# {{{ multi_index helpers

def add_mi(mi1, mi2):
    return tuple(mi1i+mi2i for mi1i, mi2i in zip(mi1, mi2))

def mi_factorial(mi):
    from pytools import factorial
    result = 1
    for mi_i in mi:
        result *= factorial(mi_i)
    return result

def mi_power(vector, mi):
    result = 1
    for mi_i, vec_i in zip(mi, vector):
        result *= vec_i**mi_i
    return result

def mi_derivative(expr, vector, mi):
    for mi_i, vec_i in zip(mi, vector):
        expr = expr.diff(vec_i, mi_i)
    return expr


# }}}




def vector_to_device(queue, vec):
    from pytools.obj_array import with_object_array_or_scalar

    from pyopencl.array import to_device
    def to_dev(ary):
        return to_device(queue, ary)

    return with_object_array_or_scalar(to_dev, vec)
