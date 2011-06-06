#!/usr/bin/env python
#
# Here is M2M for a Taylor series:
#
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
from __future__ import division

from sympy import collect, Symbol, diff, limit, factorial, cos, ccode
from sympy.abc import x,y,a
import numpy

def moveArray(context, localName, globalName, arraySize, globalOffsetName, numThreads = 1, blockNumber = 0, localBlockNumber = 0, isLoad = True, isCoalesced = True):
  # For a shared load (like G) on the CPU, which only loads once, we can eliminate 'idx'
  if not isCoalesced:
    if not numThreads == 1:
      raise RuntimeError('Non-coalesced moves must have numThreads == 1')
    if globalOffsetName:
      globalIdx = globalOffsetName+'+'
    else:
      globalIdx = ''
    localIdx  = ''
  else:
    if globalOffsetName:
      globalIdx = globalOffsetName+'+idx+'
    else:
      globalIdx = 'idx+'
    localIdx  = 'idx+'
  numMoves  = arraySize // numThreads
  remainder = arraySize % numThreads
  if isLoad:
    context.write('// Load %s from global memory into %s in shared memory\n' % (globalName, localName))
    for i in range(numMoves):
      globalOffset = i*numThreads + blockNumber*arraySize
      localOffset  = i*numThreads + localBlockNumber*arraySize
      if localName in context['load']:
        context.write('%s[%s%d] = %s[%s%d];\n' % (localName, localIdx, localOffset, globalName, globalIdx, globalOffset))
      else:
        context.write('%s[%s%d] = 0.5;\n' % (localName, localIdx, localOffset))
    if remainder:
      globalOffset = numMoves*numThreads + blockNumber*arraySize
      localOffset  = numMoves*numThreads + localBlockNumber*arraySize
      if localName in context['load']:
        context.write('if (idx < %d) %s[%s%d] = %s[%s%d];\n' % (remainder, localName, localIdx, localOffset, globalName, globalIdx, globalOffset))
      else:
        context.write('if (idx < %d) %s[%s%d] = 0.5;\n' % (remainder, localName, localIdx, localOffset))
  else:
    if not context['store'] == 'all': return ''
    context.write('// Store %s from shared memory into %s in global memory\n' % (localName, globalName))
    for i in range(numMoves):
      globalOffset = i*numThreads + blockNumber*arraySize
      localOffset  = i*numThreads + localBlockNumber*arraySize
      context.write('%s[%s%d] = %s[%s%d];\n' % (globalName, globalIdx, globalOffset, localName, localIdx, localOffset))
    if remainder:
      globalOffset = numMoves*numThreads + blockNumber*arraySize
      localOffset  = numMoves*numThreads + localBlockNumber*arraySize
      context.write('if (idx < %d) %s[%s%d] = %s[%s%d];\n' % (remainder, globalName, globalIdx, globalOffset, localName, localIdx, localOffset))
  return ''

# Series is broken
#f = 1/(x+y)
#print f.series(x, 0, 5)
#print f.series(x, 1, 5)

def constructShiftedPolynomial(order, debug = True):
  # Start with a monomial g = \sum^{order}_{i = 0} \frac{c_i}{i!} (x - a)^i
  c = [Symbol('c'+str(i)) for i in range(order)]
  g = sum([c[i]*(x - a)**i/factorial(i) for i in range(order)])
  if debug: print g
  # Convert to a monomial
  g = collect(g.expand(), x)
  if debug: print g
  return c, g

def constructTransformMatrix(order = 5, debug = True):
  c, g = constructShiftedPolynomial(order, debug)
  # Construct matrix transform from g to \sum^{order}_{i = 0} \frac{c_i}{i!} x^i
  M = []
  for o in range(order):
    exp = g.diff(x, o).limit(x, 0)
    if debug: print exp
    M.append([exp.diff(c[p]) for p in range(order)])
  if debug: print M
  return M

def polynomialToCoefficientVector(poly, unknown, order):
  # Can we determine the order?
  return numpy.array([float(poly.diff(unknown, o).limit(unknown, 0)) for o in range(order)])

def numericalTransformMatrix(order, c, g, shift):
  M = numpy.zeros((order, order))
  for o in range(order):
    exp = g.diff(x, o).limit(x, 0).limit(a, -shift)
    for p in range(order):
      M[o,p] = exp.diff(c[p])
  print M
  return M

def testTransform(order, shift = 0.1):
  '''Test that shifted series is close to re-expansion'''
  sOrig = cos(x).series(x, 0, order)
  print 'Original series',sOrig
  #sShift = cos(x).series(x, shift, order)
  sShift = cos(x + shift).series(x, 0, order)
  print 'Shifted series',sShift
  vOrig  = polynomialToCoefficientVector(sOrig,  x, order)
  vShift = polynomialToCoefficientVector(sShift, x, order)
  print vOrig
  print vShift
  # Produce M
  c, g = constructShiftedPolynomial(order)
  M    =  numericalTransformMatrix(order, c, g, shift)
  # Shift series
  vShiftComp = numpy.dot(M, vOrig)
  print vShiftComp
  print vShift - vShiftComp
  assert(numpy.linalg.norm(vShift - vShiftComp, 2) < 1.0e-2)
  return

def generateM2M(dim, order, numThreads, debug = False):
  '''Output an OpenCL code for this operation
     1) Fully unrolled using shift 't'
     2) Loops using shift 't'
     3) Substitute 'level' for 't' '''
  from mako.template import Template
  M    = constructTransformMatrix(order, False)
  # Given M, let each thread do matvec
  source = Template('''
<%namespace name="fm" module="m2m"/>
__kernel void M2M(__global float a, __global const float *child, __global float *parent) {
  int   parentOffset = get_group_id(0) * ${order*numThreads};
  int   childOffset  = get_group_id(0) * ${order*numThreads*2**dim};
  int   idx          = get_local_id(0);
  __local float locParent[${order*numThreads}];
  __local float locChild[${order*numThreads*2**dim}];

  ${fm.moveArray('locChild', 'child', order*numThreads*2**dim, 'childOffset', numThreads = numThreads, isLoad = True)}
  % for o in range(order):
  locParent[idx*${order}+${o}] = 0.0;
  % endfor
  % for c in range(2**dim):
  %   for o in range(order):
  locParent[idx*${order}+${o}] += ${' + '.join(['(%s)*locChild[%d]' % (ccode(M[o][p]), c*order+p) for p in range(order)])};
  %   endfor
  % endfor
  //printf("locParent %g %g %g %g %g\\n", locParent[0], locParent[1], locParent[2], locParent[3], locParent[4]);
  ${fm.moveArray('locParent', 'parent', order*numThreads, 'parentOffset', numThreads = numThreads, isLoad = False)}
}
''')
  code = source.render(dim=dim, order=order, numThreads=numThreads, M=M, ccode=ccode, store='all', load=['locParent', 'locChild'])
  if debug: print code
  # Output flop and byte count for kernel
  # Quantify accuracy
  return code

def test_m2m(debug = False):
  import pyopencl as cl
  import pyopencl.array as cl_array
  import numpy as np

  if debug: print cl.get_platforms()[0].get_devices()
  dev = cl.get_platforms()[0].get_devices()[1]
  if debug: print dev.name
  ctx = cl.Context([dev])
  queue = cl.CommandQueue(ctx)

  dim   = 1
  order = 5
  L     = 2
  a     = 0.1
  numChildren = 2**(dim*L)
  numParents  = 2**(dim*(L-1))
  sOrig       = cos(x).series(x, 0, order)
  vOrig       = polynomialToCoefficientVector(sOrig,  x, order)
  child       = np.zeros(numChildren*order).astype(np.float32)
  for c in range(numChildren):
    child[order*c:order*(c+1)] = vOrig
  child_dev   = cl_array.to_device(queue, child)

  numThreads  = 1
  kernel_text = generateM2M(dim, order, numThreads)
  prg = cl.Program(ctx, kernel_text).build()

  M2M = prg.M2M
  M2M.set_scalar_arg_dtypes([np.float32, None, None])

  parent_dev = cl_array.empty(queue, numParents * order, np.float32)
  if debug: print (numParents,), (numThreads,)
  M2M(queue, (numParents,), (numThreads,), a, child_dev.data, parent_dev.data)

  parent = parent_dev.get()
  parent_host = np.empty_like(parent)
  shift = -a
  sShift = cos(x + shift).series(x, 0, order)
  vShift = polynomialToCoefficientVector(sShift, x, order)
  if debug:
    print child
    print parent
    print vShift*2.0
  for p in range(numParents):
    if debug: print parent[order*p:order*(p+1)] - vShift*2.0
    assert(numpy.linalg.norm(parent[order*p:order*(p+1)] - vShift*2.0, 2) < 1.0e-2)

  #for itarg in xrange(len(target)):
  #  potential_host[itarg] = np.sum(
  #    source[:,3]
  #    /
  #    np.sum((target[itarg,:3] - source[:,:3])**2, axis=-1)**0.5)

  #assert la.norm(potential - potential_host)/la.norm(potential_host) < 1e-3
  return True

if __name__ == '__main__':
  #testTransform(5)
  print test_m2m()
