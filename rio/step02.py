import numpy as np
N = 10
eps2 = 0.0001
xi = -np.random.rand(N)
yi = -np.random.rand(N)
zi = -np.random.rand(N)
pi = np.zeros(N)
xj = np.random.rand(N)
yj = np.random.rand(N)
zj = np.random.rand(N)
mj = np.ones(N) / N

for i in range(N):
    p = 0
    for j in range(N):
        dx = xi[i] - xj[j]
        dy = yi[i] - yj[j]
        dz = zi[i] - zj[j]
        r = np.sqrt(dx * dx + dy * dy + dz * dz + eps2)
        p += mj[j] / r
    pi[i] = p

xc = yc = zc = 0.5
multipole = np.zeros(10)
for j in range(N):
    dx = xc - xj[j]
    dy = yc - yj[j]
    dz = zc - zj[j]
    multipole[0] += mj[j]
    multipole[1] += mj[j] * dx
    multipole[2] += mj[j] * dy
    multipole[3] += mj[j] * dz
    multipole[4] += mj[j] * dx * dx / 2
    multipole[5] += mj[j] * dy * dy / 2
    multipole[6] += mj[j] * dz * dz / 2
    multipole[7] += mj[j] * dx * dy / 2
    multipole[8] += mj[j] * dy * dz / 2
    multipole[9] += mj[j] * dz * dx / 2

err = rel = 0
for i in range(N):
    p = 0
    X = xi[i] - xc
    Y = yi[i] - yc
    Z = zi[i] - zc
    R = np.sqrt(X * X + Y * Y + Z * Z)
    R3 = R * R * R
    R5 = R3 * R * R
    p += multipole[0] / R
    p += multipole[1] * (-X / R3)
    p += multipole[2] * (-Y / R3)
    p += multipole[3] * (-Z / R3)
    p += multipole[4] * (3 * X * X / R5 - 1 / R3)
    p += multipole[5] * (3 * Y * Y / R5 - 1 / R3)
    p += multipole[6] * (3 * Z * Z / R5 - 1 / R3)
    p += multipole[7] * (3 * X * Y / R5)
    p += multipole[8] * (3 * Y * Z / R5)
    p += multipole[9] * (3 * Z * X / R5)
    err += (pi[i] - p) * (pi[i] - p)
    rel += pi[i] * pi[i]
    print i, pi[i], p
print 'error : ', np.sqrt(err/rel)
