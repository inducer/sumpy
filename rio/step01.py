import numpy as np
N = 10
eps2 = 0.0001
x = np.random.rand(N)
y = np.random.rand(N)
z = np.random.rand(N)
m = np.ones(N) / N
for i in range(N):
    p = -m[i] / np.sqrt(eps2)
    for j in range(N):
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dz = z[i] - z[j]
        r = np.sqrt(dx * dx + dy * dy + dz * dz + eps2)
        p += m[j] / r
    print i, p
