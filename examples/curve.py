import numpy as np
import scipy as sp
import scipy.fftpack


class CurveGrid:
    def __init__(self, x, y):
        self.pos = np.vstack([x, y]).copy()
        xp = self.xp = sp.fftpack.diff(x, period=1)
        yp = self.yp = sp.fftpack.diff(y, period=1)
        xpp = self.xpp = sp.fftpack.diff(xp, period=1)
        ypp = self.ypp = sp.fftpack.diff(yp, period=1)
        self.mean_curvature = (xp*ypp-yp*xpp)/((xp**2+yp**2)**(3/2))

        speed = self.speed = np.sqrt(xp**2+yp**2)
        self.normal = (np.vstack([yp, -xp])/speed).copy()

    def __len__(self):
        return len(self.pos)

    def plot(self):
        import matplotlib.pyplot as pt
        pt.plot(self.pos[:, 0], self.pos[:, 1])
