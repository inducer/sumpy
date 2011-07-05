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

xc = yc = zc = np.zeros(9)
multipole = np.zeros((9,10))
for i in range(8):
    xc[i] = ( i      % 2) * 0.5 + 0.25
    yc[i] = ((i / 2) % 2) * 0.5 + 0.25
    zc[i] = ( i / 4     ) * 0.5 + 0.25
xc[8] = yc[8] = zc[8] = 0.5

for j in range(N):
    i = (xj[j] > xc[8]) + ((yj[j] > yc[8]) << 1) + ((zj[j] > zc[8]) << 2)
    dx = xc[i] - xj[j]
    dy = yc[i] - yj[j]
    dz = zc[i] - zj[j]
    multipole[i][0] += mj[j]
    multipole[i][1] += mj[j] * dx
    multipole[i][2] += mj[j] * dy
    multipole[i][3] += mj[j] * dz
    multipole[i][4] += mj[j] * dx * dx / 2
    multipole[i][5] += mj[j] * dy * dy / 2
    multipole[i][6] += mj[j] * dz * dz / 2
    multipole[i][7] += mj[j] * dx * dy / 2
    multipole[i][8] += mj[j] * dy * dz / 2
    multipole[i][9] += mj[j] * dz * dx / 2

for i in range(8):
    dx = xc[8] - xc[i]
    dy = yc[8] - yc[i]
    dz = zc[8] - zc[i]
    multipole[8][0] += multipole[i][0]
    multipole[8][1] += multipole[i][1] +  dx*multipole[i][0]
    multipole[8][2] += multipole[i][2] +  dy*multipole[i][0]
    multipole[8][3] += multipole[i][3] +  dz*multipole[i][0]
    multipole[8][4] += multipole[i][4] +  dx*multipole[i][1] + dx * dx * multipole[i][0] / 2
    multipole[8][5] += multipole[i][5] +  dy*multipole[i][2] + dy * dy * multipole[i][0] / 2
    multipole[8][6] += multipole[i][6] +  dz*multipole[i][3] + dz * dz * multipole[i][0] / 2
    multipole[8][7] += multipole[i][7] + (dx*multipole[i][2] + dy * multipole[i][1] + dx * dy * multipole[i][0]) / 2
    multipole[8][8] += multipole[i][8] + (dy*multipole[i][3] + dz * multipole[i][2] + dy * dz * multipole[i][0]) / 2
    multipole[8][9] += multipole[i][9] + (dz*multipole[i][1] + dx * multipole[i][3] + dz * dx * multipole[i][0]) / 2

err = rel = 0
for i in range(N):
    p = 0
    X = xi[i] - xc[8]
    Y = yi[i] - yc[8]
    Z = zi[i] - zc[8]
    R = np.sqrt(X * X + Y * Y + Z * Z)
    R3 = R * R * R
    R5 = R3 * R * R
    p += multipole[8][0] / R
    p += multipole[8][1] * (-X / R3)
    p += multipole[8][2] * (-Y / R3)
    p += multipole[8][3] * (-Z / R3)
    p += multipole[8][4] * (3 * X * X / R5 - 1 / R3)
    p += multipole[8][5] * (3 * Y * Y / R5 - 1 / R3)
    p += multipole[8][6] * (3 * Z * Z / R5 - 1 / R3)
    p += multipole[8][7] * (3 * X * Y / R5)
    p += multipole[8][8] * (3 * Y * Z / R5)
    p += multipole[8][9] * (3 * Z * X / R5)
    err += (pi[i] - p) * (pi[i] - p)
    rel += pi[i] * pi[i]
    print i, pi[i], p
print 'error : ', np.sqrt(err/rel)
