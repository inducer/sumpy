__copyright__ = "Copyright (C) 2022 Alexandru Fikl"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

__doc__ = """
Array Context
-------------

.. autofunction:: make_loopy_program
.. autoclass:: PyOpenCLArrayContext
"""


from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytato as pt
import sumpy.transform.metadata as mtd

from boxtree.array_context import (
    PyOpenCLArrayContext as PyOpenCLArrayContextBase,
    PytatoPyOpenCLArrayContext as PytatoPyOpenCLArrayContextBase
)
from arraycontext.pytest import (
        _PytestPyOpenCLArrayContextFactoryWithClass,
        register_pytest_array_context_factory)
from pytools.tag import ToTagSetConvertible



# {{{ PyOpenCLArrayContext

class PyOpenCLArrayContext(PyOpenCLArrayContextBase):
    def transform_loopy_program(self, t_unit):

        knl = t_unit.default_entrypoint

        return t_unit

# }}}


# {{{ PytatoPyOpenCLArrayContext

class PytatoPyOpenCLArrayContext(PytatoPyOpenCLArrayContextBase):

    def transform_dag(self, dag):
        try:
            self.dot_codes.append(pt.get_dot(dag))
        except AttributeError:
            self.dot_codes = []
            self.dot_codes.append(pt.get_dot_graph(dag))

        return super().transform_dag(dag)


    def call_loopy(self, program, **kwargs):

        import loopy as lp

        # {{{ preprocess arguments

        knl = program.default_entrypoint

        # shape inference
        new_args = []
        for arg in knl.args:
            new_arg = arg
            if isinstance(arg, lp.ArrayArg) and arg.shape is None:
                new_arg = new_arg.copy(shape=kwargs[arg.name].shape)

            new_args.append(new_arg)

        knl = knl.copy(args=new_args)
        program = program.with_kernel(knl)

        # }}}

        # {{{ preprocess kwargs

        # remove unnecessary kwargs
        if "wait_for" in kwargs.keys():
            kwargs.pop("wait_for")

        # remove output args from kwargs (they are implicit in CallLoopy nodes)
        for arg in program.default_entrypoint.args:
            if arg.is_output and arg.name in kwargs.keys():
                kwargs.pop(arg.name)

        # }}}

        return super().call_loopy(program, **kwargs)


    def transform_loopy_program(self, t_unit):

        knl = t_unit.default_entrypoint

        return t_unit

# }}}


def make_loopy_program(
        domains, statements,
        kernel_data: Optional[List[Any]] = None, *,
        name: str = "sumpy_loopy_kernel",
        silenced_warnings: Optional[Union[List[str], str]] = None,
        assumptions: Optional[Union[List[str], str]] = None,
        fixed_parameters: Optional[Dict[str, Any]] = None,
        index_dtype: Optional["np.dtype"] = None,
        tags: ToTagSetConvertible = None):
    """Return a :class:`loopy.LoopKernel` suitable for use with
    :meth:`arraycontext.ArrayContext.call_loopy`.
    """
    if kernel_data is None:
        kernel_data = [...]

    if silenced_warnings is None:
        silenced_warnings = []

    import loopy as lp
    from arraycontext.loopy import _DEFAULT_LOOPY_OPTIONS

    return lp.make_kernel(
            domains,
            statements,
            kernel_data=kernel_data,
            options=_DEFAULT_LOOPY_OPTIONS,
            default_offset=lp.auto,
            name=name,
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
            assumptions=assumptions,
            fixed_parameters=fixed_parameters,
            silenced_warnings=silenced_warnings,
            index_dtype=index_dtype,
            tags=tags)


def is_cl_cpu(actx: PyOpenCLArrayContext) -> bool:
    import pyopencl as cl
    return all(dev.type & cl.device_type.CPU for dev in actx.context.devices)


# {{{ pytest

def _acf():
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    return PyOpenCLArrayContext(queue, force_device_scalars=True)


class PytestPyOpenCLArrayContextFactory(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class = PyOpenCLArrayContext

    def __call__(self):
        # NOTE: prevent any cache explosions during testing!
        from sympy.core.cache import clear_cache
        clear_cache()

        return super().__call__()


register_pytest_array_context_factory(
    "sumpy.pyopencl",
    PytestPyOpenCLArrayContextFactory)

# }}}
