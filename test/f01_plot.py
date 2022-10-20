import matplotlib.pyplot as plt
from OneLoopLFVHD.LFVHDFeynG_mpmath2 import R0
import numpy as np
import mpmath as mp
mp.dps = 500

#y = np.linspace(-1,2,100)
#z0 = np.array([abs(f0np(yi)) for yi in y])
#z1 = np.array([abs(f1np(yi)) for yi in y])
#plt.figure()
#plt.semilogy(y,z0,'.')
#plt.semilogy(y,z1,'.')
#plt.show()

# @jit
def x0(ma, M0, M2,numeric=True):
    '''
    x0 root

    Parameters
    ----------
    ma: float, mpf
    Mass of the Higgs H_a

    M0: float, mpf
    Mass of P0 particle inside the loop

    M2: float, mpf
    Mass of P2 particle inside the loop
    '''
    if numeric:
        out = (M2**2 - M0**2)/ma**2
    else:
        out = ((M2 - M0)*(M2 + M0))/ma**2
    return out


# @jit
def x3(M0, M1, numeric=True):
    '''
    x3 root

    Parameters
    ----------

    M0: float, mpf
    Mass of P0 particle inside the loop

    M1: float, mpf
    Mass of P1 particle inside the loop
    '''
    if numeric:
        out = (-M0**2)/(M1**2 - M0**2)
    else:
        out = (-M0**2)/((M1 - M0)*(M1 + M0))
    return out

n = 100
expmp = mp.linspace(-1,15,n)
mn = np.array([mp.mpf('10.0')**k for k in expmp])
mw = 80.739
mh = 125.1

plt.figure()
x0T = np.array([abs(x0(mh,mw,m)) for m in mn])
# print(x0T)
x0F = np.array([abs(x0(mh,mw,m,False)) for m in mn])
plt.loglog(mn,x0T,'.')
plt.loglog(mn,x0F,'.')
plt.title('$x_0$',fontsize=18)
plt.xlabel('$m_n$',fontsize=18)
#plt.show()


plt.figure()
x3T = np.array([abs(x3(mw,m)) for m in mn])
# print(x0T)
x3F = np.array([abs(x3(mw,m,False)) for m in mn])
plt.loglog(mn,x3T,'.')
plt.loglog(mn,x3F,'.')
plt.title('$x_3$',fontsize=18)
plt.xlabel('$m_n$',fontsize=18)
#plt.show()


r0t = []
for x0t, x3t in zip(x0T,x3T):
    r0t.append(abs(R0(x0t,x3t)))
R0T = np.array(r0t)

r0f = []
for x0f, x3f in zip(x0F,x3F):
    r0f.append(abs(R0(x0f,x3f)))
R0F = np.array(r0f)

plt.figure()
plt.loglog(mn,R0T,'.')
#plt.loglog(mn,R0F,'.')
plt.title('$R_0$',fontsize=18)
plt.xlabel('$m_n$',fontsize=18)
plt.show()