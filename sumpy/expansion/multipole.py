from __future__ import division

import sympy as sp
from pytools import memoize_method




# {{{ base class

class MultipoleExpansionBase(object):
    def __init__(self, kernel, order
