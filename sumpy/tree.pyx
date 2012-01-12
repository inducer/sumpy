cimport numpy as np
import numpy as np


cdef struct Leaf:
    cdef int I
    cdef float X[3]
    cdef Leaf *NEXT

cdef struct Node:
    cdef int NOCHILD
    cdef int LEVEL
    cdef int NLEAF
    cdef int CHILD[8]
    cdef float X[3]
    cdef int LEAF

cdef struct Cell:
    cdef int ICELL
    cdef int NCHILD
    cdef int NCLEAF
    cdef int NDLEAF
    cdef int PARENT
    cdef int CHILD
    cdef int LEAF
    cdef float X[3] # box center
    cdef float R
    cdef float RCRIT




ctypedef np.float64_t COORDINATE_t

class TopDown:
    cdef init(Node &node):
        node.NOCHILD = 1
        node.NLEAF = 0
        node.LEAF = 0
        for b in xrange(8):
            node.CHILD[b] = -1

    cdef addChild(nodes, int octant, Node *N):
        cdef Node child
        init(child)
        child.LEVEL = N->LEVEL+1
        cdef float r = R0 / (1 << child.LEVEL)
        for d in xrange(3):
            child.X[d] += r * (((octant & 1 << d) >> d) * 2 - 1)
        N->NOCHILD = False
        N->CHILD[octant] = len(nodes)
        nodes.append(child)
        NCELL += 1

     cdef splitNode(self, Node *N):
         while N->NLEAF > NCRIT:
             int c = 0
             Leaf *Ln
             for( Leaf *L=N->LEAF; L; L=Ln )
                 Ln = L->NEXT
                 int octant = (L->X[0] > N->X[0]) + ((L->X[1] > N->X[1]) << 1) + ((L->X[2] > N->X[2]) << 2);
                 if N->CHILD[octant] == -1:
                     addChild(octant,N);

                 c = N->CHILD[octant];
                 Node *child = &nodes[c];
                 L->NEXT = child->LEAF;
                 child->LEAF = L;
                 child->NLEAF++;

             N = nodes.begin()+c;

      cdef nodes2cells(int i, C_iter C) {
          C->R      = R0 / (1 << nodes[i].LEVEL);
          C->X      = nodes[i].X;
          C->NDLEAF = nodes[i].NLEAF;
          C->LEAF   = BN;
          if( nodes[i].NOCHILD ) {
              C->CHILD = 0;
              C->NCHILD = 0;
              C->NCLEAF = nodes[i].NLEAF;
              for( Leaf *L=nodes[i].LEAF; L; L=L->NEXT ) {
                  BN->IBODY = L->I;
                  BN++;
              }
              C->NCLEAF = 0;
              int nsub=0;
              for( int octant=0; octant!=8; ++octant ) {
                if( nodes[i].CHILD[octant] != -1 ) {
                   ++nsub;
                                                                                                                                                                                                }
          }
          C_iter Ci = CN;
          C->CHILD = Ci - Ci0;
          C->NCHILD = nsub;
          CN += nsub;
          for( int octant=0; octant!=8; ++octant ) {
            if( nodes[i].CHILD[octant] != -1 ) {
          Ci->PARENT = C - Ci0;
              nodes2cells(nodes[i].CHILD[octant], Ci++);
            }
          }
        }


    cdef build_tree(np.ndarray[COORDINATE_t, ndim=2] points):

        npoints, ndim = points.shape

