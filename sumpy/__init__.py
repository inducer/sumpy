from __future__ import annotations


__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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

import os
from typing import TYPE_CHECKING

from pytools.persistent_dict import WriteOncePersistentDict

from sumpy.e2e import (
    E2EFromChildren,
    E2EFromCSR,
    E2EFromParent,
    M2LGenerateTranslationClassesDependentData,
    M2LPostprocessLocal,
    M2LPreprocessMultipole,
    M2LUsingTranslationClassesDependentData,
)
from sumpy.e2p import E2PFromCSR, E2PFromSingleBox
from sumpy.p2e import P2EFromCSR, P2EFromSingleBox
from sumpy.p2p import P2P, P2PFromCSR
from sumpy.version import VERSION_TEXT


if TYPE_CHECKING:
    from collections.abc import Hashable

    import loopy as lp


__all__ = [
    "P2P",
    "E2EFromCSR",
    "E2EFromChildren",
    "E2EFromParent",
    "E2PFromCSR",
    "E2PFromSingleBox",
    "M2LGenerateTranslationClassesDependentData",
    "M2LPostprocessLocal",
    "M2LPreprocessMultipole",
    "M2LUsingTranslationClassesDependentData",
    "P2EFromCSR",
    "P2EFromSingleBox",
    "P2PFromCSR",
]


code_cache: WriteOncePersistentDict[Hashable, lp.TranslationUnit] = (
    WriteOncePersistentDict(f"sumpy-code-cache-v8-{VERSION_TEXT}", safe_sync=False))


# {{{ optimization control

OPT_ENABLED = True

OPT_ENABLED = "SUMPY_NO_OPT" not in os.environ


def set_optimization_enabled(flag):
    """Set whether the :mod:`loopy` kernels should be optimized."""
    global OPT_ENABLED
    OPT_ENABLED = flag

# }}}


# {{{ cache control

CACHING_ENABLED = (
    "SUMPY_NO_CACHE" not in os.environ
    and "CG_NO_CACHE" not in os.environ)

NO_CACHE_KERNELS = tuple(os.environ.get("SUMPY_NO_CACHE_KERNELS",
                                        "").split(","))


def set_caching_enabled(flag, no_cache_kernels=()):
    """Set whether :mod:`loopy` is allowed to use disk caching for its various
    code generation stages.
    """
    global CACHING_ENABLED, NO_CACHE_KERNELS
    NO_CACHE_KERNELS = no_cache_kernels
    CACHING_ENABLED = flag


class CacheMode:
    """A context manager for setting whether :mod:`sumpy` is allowed to use
    disk caches.
    """

    def __init__(self, new_flag, new_no_cache_kernels=()):
        self.new_flag = new_flag
        self.new_no_cache_kernels = new_no_cache_kernels

    def __enter__(self):
        global CACHING_ENABLED, NO_CACHE_KERNELS
        self.previous_flag = CACHING_ENABLED
        self.previous_kernels = NO_CACHE_KERNELS
        set_caching_enabled(self.new_flag, self.new_no_cache_kernels)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_caching_enabled(self.previous_flag, self.previous_kernels)
        del self.previous_flag
        del self.previous_kernels

# }}}
