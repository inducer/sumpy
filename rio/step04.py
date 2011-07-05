import numpy as np
NCRIT = 10

class cell:
    def __init__(self):
        self.nleaf = self.nchild = None
        self.leaf = np.zeros(NCRIT).astype(np.int)
        self.parent = None
        self.child = np.zeros(8).astype(np.int)
        self.xc = self.yc = self.zc = self.r = None

def initialize(C,i):
    C[i].nleaf = C[i].nchild = 0
    C[i].parent = 0
    for c in range(8): C[i].child[c] = 0

def add_child(octant, C, i, ic):
    ic += 1
    initialize(C,ic)
    C[ic].r  = C[i].r / 2
    C[ic].xc = C[i].xc + C[ic].r * ((octant & 1) * 2 - 1)
    C[ic].yc = C[i].yc + C[ic].r * ((octant & 2) - 1    )
    C[ic].zc = C[i].zc + C[ic].r * ((octant & 4) / 2 - 1)
    C[ic].parent = i
    C[i].child[octant] = ic
    C[i].nchild |= (1 << octant)
    return ic

def split_cell(x, y, z, C, i, ic):
    for l in range(NCRIT):
        ib = C[i].leaf[l]
        octant = (x[ib] > C[i].xc) + ((y[ib] > C[i].yc) << 1) + ((z[ib] > C[i].zc) << 2)
        if( not(C[i].nchild & (1 << octant)) ): ic = add_child(octant,C,i,ic)
        ii = C[i].child[octant]
        C[ii].leaf[C[ii].nleaf] = ib
        C[ii].nleaf += 1
        if( C[ii].nleaf >= NCRIT ): ic = split_cell(x,y,z,C,ii,ic)
    return ic

def traverse(C,i,cells,leafs):
    if( C[i].nleaf >= NCRIT ):
        for c in range(8):
            if( C[i].nchild & (1 << c) ): cells,leafs = traverse(C,C[i].child[c],cells,leafs)
    else:
        for l in range(C[i].nleaf):
            print cells,leafs,C[i].leaf[l]
            leafs += 1
        cells += 1
    return cells,leafs

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
z = np.random.rand(N)

C = [cell() for i in range(N)]
initialize(C,0)
C[0].xc = C[0].yc = C[0].zc = C[0].r = 0.5
ic = 0
for ib in range(N):
    i = 0
    while( C[i].nleaf >= NCRIT ):
        C[i].nleaf += 1
        octant = (x[ib] > C[i].xc) + ((y[ib] > C[i].yc) << 1) + ((z[ib] > C[i].zc) << 2)
        if( not(C[i].nchild & (1 << octant)) ): ic = add_child(octant,C,i,ic)
        i = C[i].child[octant]
    C[i].leaf[C[i].nleaf] = ib
    C[i].nleaf += 1
    if( C[i].nleaf >= NCRIT ): ic = split_cell(x,y,z,C,i,ic)

cells = leafs = 0
cells,leafs = traverse(C,0,cells,leafs);
