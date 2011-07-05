import numpy as np
NCRIT = 10
THETA = 0.5
EPS2 = 0.0001

class cell:
    def __init__(self):
        self.nleaf = self.nchild = None
        self.leaf = np.zeros(NCRIT).astype(np.int)
        self.parent = None
        self.child = np.zeros(8).astype(np.int)
        self.xc = self.yc = self.zc = self.r = None
        self.multipole = np.zeros(10)

def initialize(C,i):
    C[i].nleaf = C[i].nchild = 0
    C[i].parent = 0
    for c in range(8): C[i].child[c] = 0
    for m in range(10): C[i].multipole[m] = 0

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

def getMultipole(C, i, x, y, z, m, twig, ntwig):
    if( C[i].nleaf >= NCRIT ):
        for c in range(8):
            if( C[i].nchild & (1 << c) ): twig,ntwig = getMultipole(C,C[i].child[c],x,y,z,m,twig,ntwig)
    else:
        for l in range(C[i].nleaf):
            j = C[i].leaf[l]
            dx = C[i].xc - x[j]
            dy = C[i].yc - y[j]
            dz = C[i].zc - z[j]
            C[i].multipole[0] += m[j];
            C[i].multipole[1] += m[j] * dx;
            C[i].multipole[2] += m[j] * dy;
            C[i].multipole[3] += m[j] * dz;
            C[i].multipole[4] += m[j] * dx * dx / 2;
            C[i].multipole[5] += m[j] * dy * dy / 2;
            C[i].multipole[6] += m[j] * dz * dz / 2;
            C[i].multipole[7] += m[j] * dx * dy / 2;
            C[i].multipole[8] += m[j] * dy * dz / 2;
            C[i].multipole[9] += m[j] * dz * dx / 2;
        twig[ntwig] = i
        ntwig += 1
    return twig,ntwig

def upwardSweep(C, i, ip):
    dx = C[ip].xc - C[i].xc;
    dy = C[ip].yc - C[i].yc
    dz = C[ip].zc - C[i].zc
    C[ip].multipole[0] += C[i].multipole[0]
    C[ip].multipole[1] += C[i].multipole[1] +  dx*C[i].multipole[0]
    C[ip].multipole[2] += C[i].multipole[2] +  dy*C[i].multipole[0]
    C[ip].multipole[3] += C[i].multipole[3] +  dz*C[i].multipole[0]
    C[ip].multipole[4] += C[i].multipole[4] +  dx*C[i].multipole[1] + dx * dx * C[i].multipole[0] / 2
    C[ip].multipole[5] += C[i].multipole[5] +  dy*C[i].multipole[2] + dy * dy * C[i].multipole[0] / 2
    C[ip].multipole[6] += C[i].multipole[6] +  dz*C[i].multipole[3] + dz * dz * C[i].multipole[0] / 2
    C[ip].multipole[7] += C[i].multipole[7] + (dx*C[i].multipole[2] +      dy * C[i].multipole[1] + dx * dy * C[i].multipole[0]) / 2
    C[ip].multipole[8] += C[i].multipole[8] + (dy*C[i].multipole[3] +      dz * C[i].multipole[2] + dy * dz * C[i].multipole[0]) / 2
    C[ip].multipole[9] += C[i].multipole[9] + (dz*C[i].multipole[1] +      dx * C[i].multipole[3] + dz * dx * C[i].multipole[0]) / 2

def evaluate(C, i, j, x, y, z, m, p):
    if( C[j].nleaf >= NCRIT ):
        for c in range(8):
            if( C[j].nchild & (1 << c) ):
                jj = C[j].child[c]
                dx = C[i].xc - C[jj].xc
                dy = C[i].yc - C[jj].yc
                dz = C[i].zc - C[jj].zc
                r = np.sqrt(dx * dx + dy * dy + dz * dz);
                if( C[i].r+C[jj].r > THETA*r ):
                    p = evaluate(C,i,jj,x,y,z,m,p)
                else:
                    for l in range(C[i].nleaf):
                        ib = C[i].leaf[l]
                        X = x[ib] - C[jj].xc
                        Y = y[ib] - C[jj].yc
                        Z = z[ib] - C[jj].zc
                        R = np.sqrt(X * X + Y * Y + Z * Z)
                        R3 = R * R * R
                        R5 = R3 * R * R
                        p[ib] += C[jj].multipole[0] / R
                        p[ib] += C[jj].multipole[1] * (-X / R3)
                        p[ib] += C[jj].multipole[2] * (-Y / R3)
                        p[ib] += C[jj].multipole[3] * (-Z / R3)
                        p[ib] += C[jj].multipole[4] * (3 * X * X / R5 - 1 / R3)
                        p[ib] += C[jj].multipole[5] * (3 * Y * Y / R5 - 1 / R3)
                        p[ib] += C[jj].multipole[6] * (3 * Z * Z / R5 - 1 / R3)
                        p[ib] += C[jj].multipole[7] * (3 * X * Y / R5)
                        p[ib] += C[jj].multipole[8] * (3 * Y * Z / R5)
                        p[ib] += C[jj].multipole[9] * (3 * Z * X / R5)
    else:
        for li in range(C[i].nleaf):
            ib = C[i].leaf[li]
            for lj in range(C[j].nleaf):
                jb = C[j].leaf[lj];
                dx = x[ib] - x[jb]
                dy = y[ib] - y[jb]
                dz = z[ib] - z[jb]
                r = np.sqrt(dx * dx + dy * dy + dz * dz + EPS2);
                p[ib] += m[jb] /r
    return p

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
z = np.random.rand(N)
m = np.ones(N) / N
p = np.zeros(N)
pd = np.zeros(N)

for i in range(N):
    pp = -m[i] / np.sqrt(EPS2)
    for j in range(N):
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dz = z[i] - z[j]
        r = np.sqrt(dx * dx + dy * dy + dz * dz + EPS2)
        pp += m[j] / r
    pd[i] = pp

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

ntwig = 0
twig = np.zeros(N).astype(np.int)
twig,ntwig = getMultipole(C,0,x,y,z,m,twig,ntwig)

for i in range(ic,0,-1):
    ip = C[i].parent
    upwardSweep(C,i,ip)

print ntwig
err = rel = 0
for i in range(N):
    p[i] = -m[i] / np.sqrt(EPS2)
for t in range(ntwig):
    i = twig[t]
    j = 0
    p = evaluate(C,i,j,x,y,z,m,p)
    for l in range(C[i].nleaf):
        ib = C[i].leaf[l]
        err += (pd[ib] - p[ib]) * (pd[ib] - p[ib])
        rel += pd[ib] * pd[ib]
        print ib, pd[ib], p[ib]
print 'error : ', np.sqrt(err/rel)
