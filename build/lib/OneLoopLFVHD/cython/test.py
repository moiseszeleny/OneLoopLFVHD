import mpmath as mp
mp.dps = 50; mp.pretty = True
from roots_c import y12
ma = 125.1
mi = 0.10566
M0 = 80.379
for i in range(1000000):
    print(y12(ma,mi,M0,mp.mpf(i)))