import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from OneLoopLFVHD.neutrinos import UpmnsStandardParametrization
from OneLoopLFVHD.neutrinos import NuOscObservables
Nudata = NuOscObservables

m1 = 1e-12  #GeV 
#current values to Square mass differences
d21 = Nudata.squareDm21.central*1e-18 # factor to convert eV^2 to GeV^2
d31 = Nudata.squareDm31.central*1e-18

#d21 = 7.5e-5*1e-18
#d31 = 2.457e-3*1e-18
m2 = np.sqrt(m1**2 + d21)
m3 = np.sqrt(m1**2 + d31)

def mndsqrt(mn1, mn2, mn3):
    return np.array(
        [
            [np.sqrt(mn1), 0, 0],
            [0, np.sqrt(mn2), 0],
            [0, 0, np.sqrt(mn3)]
        ]
        )

def angle(sin2):
    return np.arcsin(np.sqrt(sin2))

th12 = angle(Nudata.sin2theta12.central)
th13 = angle(Nudata.sin2theta13.central)
th23 = angle(Nudata.sin2theta23.central)


Upmns = UpmnsStandardParametrization(
    theta12=th12, theta13=th13, theta23=th23,
    delta=0.0, alpha1=0.0, alpha2=0.0
)

#print(Upmns)
Upmns_np = np.array([[Upmns[r, s] for r in range(3)] for s in range(3)]).T
#print(Upmns_np)

Um = Upmns_np.dot(mndsqrt(m1, m2, m3))
print(np.abs(Um[2-1, 3-1])**2)
UmmU = Um.dot(np.conjugate(Um.T))
# print(Um)

def mD_CI(MR,mux, i, j):
    return (MR**2/mux)*np.abs(Um[i, j])**2
    # return MR**2/mux*np.abs(UmmU[i, j])

n = 100
M = np.linspace(200, 1e5, n)
mux = np.linspace(1e-10, 1e-1, n)
#plt.figure()

X, Y = np.meshgrid(M, mux)

a = 2
b = 3
Z = mD_CI(X,Y, a-1, b-1)
#print(mD_CI(M,mux, 2-1, 3-1))

fig,ax=plt.subplots(1,1)
cp = ax.contourf(
    X, Y, Z,
    levels=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6,1e7,1e8],
    norm = LogNorm()
    ) # cmap=plt.cm.jet
ax.clabel(cp, inline=True, fontsize=10)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title(f'$|(m_D)_{{{a,b}}}|^2$')
ax.set_xlabel('$M$ [GeV]')
ax.set_ylabel(r'$\mu_X$ [GeV]')
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig(f'mD_contourplot_MR_mux_{a}{b}.png')
plt.show()

#plt.show()

## Upper bound erturvativity
def cota_perturvativity(mD, v):
    return np.abs(mD)**2 < 12*np.pi/v**2

v = 246
def v1(tb):
    return v*np.cos(np.arctan(tb))
    
def v2(tb):
    return v*np.sin(np.arctan(tb))
    

tb = np.logspace(-3,3)

plt.figure(figsize=(10,8))
#plt.loglog(tb, 3*np.pi*v1(tb)**2)
# plt.fill_between(tb, 3*np.pi*v1(tb)**2, alpha=0.5, label='$v_1$')
plt.loglog(tb, 3*np.pi*v2(tb)**2, )
plt.fill_between(tb, 3*np.pi*v2(tb)**2, alpha=0.5,label='$v_2$')

plt.hlines(
    y=np.abs(mD_CI(1e4, 1e-7, a-1, b-1))**2,
    xmin=1e-3, xmax=1e3, label=r'$|m_D(M_R = 10^4$ $\mu_X = 10^{-7})_{23}|^2$', linestyles='--', colors='b')

plt.hlines(
    y=np.abs(mD_CI(1e4, 1e-6, a-1, b-1))**2,
    xmin=1e-3, xmax=1e3, label=r'$|m_D(M_R = 10^4$ $\mu_X = 10^{-6})_{23}|^2$', linestyles='--', colors='g')

plt.hlines(
    y=np.abs(mD_CI(1e4, 1e-5, a-1, b-1))**2,
    xmin=1e-3, xmax=1e3, label=r'$|m_D(M_R = 10^4$ $\mu_X = 10^{-5})_{23}|^2$', linestyles='-.', colors='purple')

plt.hlines(
    y=np.abs(mD_CI(1e4, 1e-4, a-1, b-1))**2,
    xmin=1e-3, xmax=1e3, label=r'$|m_D(M_R = 10^4$ $\mu_X = 10^{-4})_{23}|^2$', linestyles='--', colors='r')

plt.hlines(
    y=np.abs(mD_CI(1e4, 1e-3, a-1, b-1))**2,
    xmin=1e-3, xmax=1e3, label=r'$|m_D(M_R = 10^4$ $\mu_X = 10^{-3})_{23}|^2$', linestyles='--', colors='y')

plt.xlabel(r'$\tan{\beta}$', fontsize=16)
plt.ylabel(r'Perturvativity bound', fontsize=16)
plt.title(r'$3 \pi v_2^2$')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(1e-3, 1e3)
plt.legend(frameon=True, ncol=2, fontsize=13)
plt.savefig('Perturvativity_bound_tb.png')
#plt.show()
########################
########################
plt.figure(figsize=(10,8))
#plt.loglog(tb, 3*np.pi*v1(tb)**2)
# plt.fill_between(tb, 3*np.pi*v1(tb)**2, alpha=0.5, label='$v_1$')
plt.loglog(tb, 3*np.pi*v2(tb)**2, )
plt.fill_between(tb, 3*np.pi*v2(tb)**2, alpha=0.5,label='$v_2$')

plt.hlines(
    y=np.abs(mD_CI(1e3, 1e-7, a-1, b-1))**2,
    xmin=1e-3, xmax=1e3, label=r'$|m_D(M_R = 10^3$ $\mu_X = 10^{-7})_{23}|^2$', linestyles='--', colors='b')

plt.hlines(
    y=np.abs(mD_CI(1e3, 1e-6, a-1, b-1))**2,
    xmin=1e-3, xmax=1e3, label=r'$|m_D(M_R = 10^3$ $\mu_X = 10^{-6})_{23}|^2$', linestyles='--', colors='g')

plt.hlines(
    y=np.abs(mD_CI(1e3, 1e-5, a-1, b-1))**2,
    xmin=1e-3, xmax=1e3, label=r'$|m_D(M_R = 10^3$ $\mu_X = 10^{-5})_{23}|^2$', linestyles='-.', colors='purple')

plt.hlines(
    y=np.abs(mD_CI(1e3, 1e-4, a-1, b-1))**2,
    xmin=1e-3, xmax=1e3, label=r'$|m_D(M_R = 10^3$ $\mu_X = 10^{-4})_{23}|^2$', linestyles='--', colors='r')

plt.hlines(
    y=np.abs(mD_CI(1e3, 1e-3, a-1, b-1))**2,
    xmin=1e-3, xmax=1e3, label=r'$|m_D(M_R = 10^3$ $\mu_X = 10^{-3})_{23}|^2$', linestyles='--', colors='y')

plt.xlabel(r'$\tan{\beta}$', fontsize=16)
plt.ylabel(r'Perturvativity bound', fontsize=16)
plt.title(r'$3 \pi v_2^2$')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(1e-3, 1e3)
plt.legend(frameon=True, ncol=2, fontsize=13)
plt.savefig('Perturvativity_bound_tb2.png')
plt.show()