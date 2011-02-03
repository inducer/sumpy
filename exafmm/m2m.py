#!/usr/bin/env python
from __future__ import division

from sympy import collect, Symbol, diff, limit, factorial, cos
from sympy.abc import x,y,a
import numpy

# Series is broken
#f = 1/(x+y)
#print f.series(x, 0, 5)
#print f.series(x, 1, 5)

#t = newCenter - center
#array[0] = 1.0
#for i in range(1, numTerms):
#  array[i*numTerms] = (-1.0/i) * pow(-t, i) # First term is logarithmic, so treat separately
#  for j in range(1, i+1)
#    array[i*numTerms+j] = pow(-t, i-j) * getBinomial(i-1, j-1)

# f(x) = f(0) + (x - 0) f'(0) + 1/2! (x - 0)^2 f''(0) + ...
#      = [f(0) f'(0) 1/2! f''(0)]
#      = [m_0 m_1 m_2]
#
# f(x) = f(a) + (x - a) f'(a) + 1/2! (x - a)^2 f''(a) + ...
#      = [f(a) f'(a) 1/2! f''(a)]
#      = [n_0 n_1 n_2]
#      = f(a) - a f'(a) + 1/2! a^2 f''(a)
#      + (x - 0) (f'(a) - 2/2! a f''(a))
#      + (x - 0)^2 1/2! f''(a)
#      = [(n_0 - a n_1 + a^2/2 n_2) (n_1 - a n_2) n_2]
#
# So
# / 1 -a a^2/2 \ n_0   m'_0
# | 0  1   -a  | n_1 = m'_1
# \ 0  0    1  / n_2   m'_2

# Start with a monomial
# y = x + a
order = 5
c = [Symbol('c'+str(i)) for i in range(order)]
g = sum([c[i]*(x - a)**i/factorial(i) for i in range(order)])
print g
g = collect(g.expand(), x)
print g
M = []
for o in range(order):
  exp = g.diff(x, o).limit(x, 0)
  print exp
  M.append([exp.diff(c[p]) for p in range(order)])
print M

shift = 0.1
# Test that shifted series is close to re-expansion
sOrig = cos(x).series(x, 0, order)
print sOrig
#sShift = cos(x).series(x, 0.1, order)
sShift = cos(x + shift).series(x, 0, order)
print sShift
# Convert original series to a vector
vOrig = [float(sOrig.diff(x, o).limit(x, 0)) for o in range(order)]
print vOrig
# Convert shifted series to a vector
vShift = [float(sShift.diff(x, o).limit(x, 0)) for o in range(order)]
print vShift
# Produce M
M = numpy.zeros((order, order))
for o in range(order):
  exp = g.diff(x, o).limit(x, 0).limit(a, -shift)
  for p in range(order):
    M[o,p] = exp.diff(c[p])
print M
# Shift series
vShiftComp = numpy.dot(M, vOrig)
print vShiftComp
print vShift - vShiftComp
assert(numpy.linalg.norm(vShift - vShiftComp, 2) < 1.0e-2)
