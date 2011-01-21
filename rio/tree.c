#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int   const NCRIT = 10;
float const THETA = 0.5;
float const EPS2  = 0.0001;

struct cell {
  int nleaf, nchild, leaf[NCRIT];
  float xc, yc, zc, r;
  float multipole[10];
  cell *parent, *child[8];
};

void initialize(cell *C) {
  C->nleaf = C->nchild = 0;
  C->parent = NULL;
  for( int i=0; i<8; i++ ) C->child[i] = NULL;
  for( int i=0; i<10; i++ ) C->multipole[i] = 0;
}

void add_child(int octant, cell *C, cell *&CN) {
  ++CN;
  initialize(CN);
  CN->r  = C->r/2;
  CN->xc = C->xc + CN->r * ((octant & 1) * 2  -1);
  CN->yc = C->yc + CN->r * ((octant & 2) - 1   );
  CN->zc = C->zc + CN->r * ((octant & 4) / 2 - 1);
  CN->parent = C;
  C->child[octant] = CN;
  C->nchild |= (1 << octant);
}

void split_cell(float *x, float *y, float *z, cell *C, cell *&CN) {
  for( int i=0; i<NCRIT; i++ ) {
    int l = C->leaf[i];
    int octant = (x[l] > C->xc) + ((y[l] > C->yc) << 1) + ((z[l] > C->zc) << 2);
    if( !(C->nchild & (1 << octant)) ) add_child(octant,C,CN);
    cell *CC = C->child[octant];
    CC->leaf[CC->nleaf++] = l;
    if( CC->nleaf >= NCRIT ) split_cell(x,y,z,CC,CN);
  }
}

void getMultipole(cell *C, float *x, float *y, float *z, float *m, cell **twig, int &ntwig) {
  float dx, dy, dz;
  if( C->nleaf >= NCRIT ) {
    for( int c=0; c<8; c++ )
      if( C->nchild & (1 << c) ) getMultipole(C->child[c],x,y,z,m,twig,ntwig);
  } else {
    for( int l=0; l<C->nleaf; l++ ) {
      int j = C->leaf[l];
      dx = C->xc-x[j];
      dy = C->yc-y[j];
      dz = C->zc-z[j];
      C->multipole[0] += m[j];
      C->multipole[1] += m[j] * dx;
      C->multipole[2] += m[j] * dy;
      C->multipole[3] += m[j] * dz;
      C->multipole[4] += m[j] * dx * dx / 2;
      C->multipole[5] += m[j] * dy * dy / 2;
      C->multipole[6] += m[j] * dz * dz / 2;
      C->multipole[7] += m[j] * dx * dy / 2;
      C->multipole[8] += m[j] * dy * dz / 2;
      C->multipole[9] += m[j] * dz * dx / 2;
    }
    twig[ntwig] = C;
    ntwig++;
  }
}

void upwardSweep(cell *C, cell *P) {
  float dx, dy, dz;
  dx = P->xc - C->xc;
  dy = P->yc - C->yc;
  dz = P->zc - C->zc;
  P->multipole[0] += C->multipole[0];
  P->multipole[1] += C->multipole[1] +  dx*C->multipole[0];
  P->multipole[2] += C->multipole[2] +  dy*C->multipole[0];
  P->multipole[3] += C->multipole[3] +  dz*C->multipole[0];
  P->multipole[4] += C->multipole[4] +  dx*C->multipole[1] + dx * dx * C->multipole[0] / 2;
  P->multipole[5] += C->multipole[5] +  dy*C->multipole[2] + dy * dy * C->multipole[0] / 2;
  P->multipole[6] += C->multipole[6] +  dz*C->multipole[3] + dz * dz * C->multipole[0] / 2;
  P->multipole[7] += C->multipole[7] + (dx*C->multipole[2] +      dy * C->multipole[1] + dx * dy * C->multipole[0]) / 2;
  P->multipole[8] += C->multipole[8] + (dy*C->multipole[3] +      dz * C->multipole[2] + dy * dz * C->multipole[0]) / 2;
  P->multipole[9] += C->multipole[9] + (dz*C->multipole[1] +      dx * C->multipole[3] + dz * dx * C->multipole[0]) / 2;
}

void evaluate(cell *CI, cell *CJ, float *x, float *y, float *z, float *m, float *p) {
  float dx, dy, dz, r, X, Y, Z, R, R3, R5;
  if( CJ->nleaf >= NCRIT ) {
    for( int c=0; c<8; c++ ) {
      if( CJ->nchild & (1 << c) ) {
        cell *CC = CJ->child[c];
        dx = CI->xc - CC->xc;
        dy = CI->yc - CC->yc;
        dz = CI->zc - CC->zc;
        r = sqrtf(dx * dx + dy * dy + dz * dz);
        if( CI->r+CC->r > THETA*r ) {
          evaluate(CI,CC,x,y,z,m,p);
        } else {
          for( int l=0; l<CI->nleaf; l++ ) {
            int i = CI->leaf[l];
            X = x[i] - CC->xc;
            Y = y[i] - CC->yc;
            Z = z[i] - CC->zc;
            R = sqrtf(X * X + Y * Y + Z * Z);
            R3 = R * R * R;
            R5 = R3 * R * R;
            p[i] += CC->multipole[0] / R;
            p[i] += CC->multipole[1] * (-X / R3);
            p[i] += CC->multipole[2] * (-Y / R3);
            p[i] += CC->multipole[3] * (-Z / R3);
            p[i] += CC->multipole[4] * (3 * X * X / R5 - 1 / R3);
            p[i] += CC->multipole[5] * (3 * Y * Y / R5 - 1 / R3);
            p[i] += CC->multipole[6] * (3 * Z * Z / R5 - 1 / R3);
            p[i] += CC->multipole[7] * (3 * X * Y / R5);
            p[i] += CC->multipole[8] * (3 * Y * Z / R5);
            p[i] += CC->multipole[9] * (3 * Z * X / R5);
          }
        }
      }
    }
  } else {
    for( int li=0; li<CI->nleaf; li++ ) {
      int i = CI->leaf[li];
      for( int lj=0; lj<CJ->nleaf; lj++ ) {
        int j = CJ->leaf[lj];
        dx = x[i] - x[j];
        dy = y[i] - y[j];
        dz = z[i] - z[j];
        r = sqrtf(dx * dx + dy * dy + dz * dz + EPS2);
        p[i] += m[j] / r;
      }
    }
  }
}

int main() {
  int N = 50;
  float x[N], y[N], z[N], m[N], p[N], pd[N];
// Initialize
  for( int i=0; i<N; i++ ) {
    x[i] = rand() / (1. + RAND_MAX) * 2 - 1;
    y[i] = rand() / (1. + RAND_MAX) * 2 - 1;
    z[i] = rand() / (1. + RAND_MAX) * 2 - 1;
    m[i] = 1.0 / N;
  }
// Direct summation
  float dx, dy, dz, r;
  for( int i=0; i<N; i++ ) {
    float pp = - m[i] / sqrtf(EPS2);
    for( int j=0; j<N; j++ ) {
      dx = x[i] - x[j];
      dy = y[i] - y[j];
      dz = z[i] - z[j];
      r = sqrtf(dx * dx + dy * dy + dz * dz + EPS2);
      pp += m[j] / r;
    }
    pd[i] = pp;
  }

// Set root cell
  cell C0[N];
  initialize(C0);
  C0->xc = C0->yc = C0->zc = C0->r = 0.5;
// Build tree
  cell *CN = C0;
  for( int i=0; i<N; i++ ) {
    cell *C = C0;
    while( C->nleaf >= NCRIT ) {
      C->nleaf++;
      int octant = (x[i] > C->xc) + ((y[i] > C->yc) << 1) + ((z[i] > C->zc) << 2);
      if( !(C->nchild & (1 << octant)) ) add_child(octant,C,CN);
      C = C->child[octant];
    }
    C->leaf[C->nleaf++] = i;
    if( C->nleaf >= NCRIT ) split_cell(x,y,z,C,CN);
  }
// Multipole expansion
  int ntwig=0;
  cell *twig[N];
  getMultipole(C0,x,y,z,m,twig,ntwig);
// Upward translation
  for( cell *C=CN; C!=C0; --C ) {
    cell *P = C->parent;
    upwardSweep(C,P);
  }
// Evaluate expansion
  float err=0,rel=0;
  for( int i=0; i<N; i++ ) p[i] = -m[i] / sqrtf(EPS2);
  for( int t=0; t<ntwig; t++ ) {
    cell *CI = twig[t];
    cell *CJ = C0;
    evaluate(CI,CJ,x,y,z,m,p);
    for( int l=0; l<CI->nleaf; l++ ) {
      int i = CI->leaf[l];
      err += (pd[i] - p[i]) * (pd[i] - p[i]);
      rel += pd[i] * pd[i];
      printf("%d %f %f\n",i,pd[i],p[i]);
    }
  }
  printf("error : %f\n",sqrtf(err/rel));
}
