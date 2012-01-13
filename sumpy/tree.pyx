import numpy as np
cimport numpy as np

particles_per_box = 10

class Cell:
    def __init__(self):
        self.nleaf = self.nchild = None
        self.leaf = np.zeros(particles_per_box).astype(np.int)
        self.parent = None
        self.child = np.zeros(8).astype(np.int)
        self.xc = self.yc = self.zc = self.r = None
        self.multipole = np.zeros(10)

def initialize(cells,i):
    cells[i].nleaf = cells[i].nchild = 0
    cells[i].parent = 0
    for c in range(8): cells[i].child[c] = 0
    for m in range(10): cells[i].multipole[m] = 0

def add_child(octant, cells, i, ic):
    ic += 1
    initialize(cells,ic)
    cells[ic].r  = cells[i].r / 2
    cells[ic].xc = cells[i].xc + cells[ic].r * ((octant & 1) * 2 - 1)
    cells[ic].yc = cells[i].yc + cells[ic].r * ((octant & 2) - 1    )
    cells[ic].zc = cells[i].zc + cells[ic].r * ((octant & 4) / 2 - 1)
    cells[ic].parent = i
    cells[i].child[octant] = ic
    cells[i].nchild |= (1 << octant)
    return ic

def split_cell(x, y, z, cells, i, ic):
    for l in range(particles_per_box):
        ib = cells[i].leaf[l]

        octant = (
                (x[ib] > cells[i].xc)
                + ((y[ib] > cells[i].yc) << 1)
                + ((z[ib] > cells[i].zc) << 2))

        if not (cells[i].nchild & (1 << octant)):
            ic = add_child(octant,cells,i,ic)

        ii = cells[i].child[octant]
        cells[ii].leaf[cells[ii].nleaf] = ib
        cells[ii].nleaf += 1
        if cells[ii].nleaf >= particles_per_box:
            ic = split_cell(x,y,z,cells,ii,ic)

    return ic

def build_tree(particles, ):
    nparticles = particles.shape[0]

    x, y, z = particles.T

    lower_left = np.min(particles, axis=0)
    upper_right = np.max(particles, axis=0)
    center = 0.5*(lower_left+upper_right)
    extent = upper_right-lower_left

    cells = [Cell() for i in range(nparticles)]
    initialize(cells,0)
    cells[0].xc, cells[0].yc, cells[0].zc = center
    cells[0].r = 0.5*max(extent)

    ic = 0
    for ib in range(nparticles):
        i = 0
        while( cells[i].nleaf >= particles_per_box ):
            cells[i].nleaf += 1
            octant = (x[ib] > cells[i].xc) + ((y[ib] > cells[i].yc) << 1) + ((z[ib] > cells[i].zc) << 2)
            if not(cells[i].nchild & (1 << octant)):
                ic = add_child(octant,cells,i,ic)

            i = cells[i].child[octant]
        cells[i].leaf[cells[i].nleaf] = ib
        cells[i].nleaf += 1

        if cells[i].nleaf >= particles_per_box:
            ic = split_cell(x,y,z,cells,i,ic)

    import matplotlib.pyplot as pt
    pt.plot(x, y, "x")


    def draw_rect(a, b):
        pts = [list(a), [a[0], b[1]], list(b), [b[0], a[1]], list(a)]
        pts = np.array(pts)
        pt.plot(pts[:,0], pts[:,1])

    ones = np.ones(2)
    for i in range(ic):
        cell = cells[i]
        center = np.array([cell.xc, cell.yc])
        draw_rect(center-cell.r*ones, center+cell.r*ones)

    #pt.savefig("x.png")
    pt.show()


