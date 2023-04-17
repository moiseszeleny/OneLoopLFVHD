from OneLoopLFVHD.neutrinos import UpmnsStandardParametrization
from OneLoopLFVHD.neutrinos import NuOscObservables
Nudata = NuOscObservables
from mpmath import mp
import numpy as np
from scipy import linalg as LAsp
LAnp = np.linalg

# ################# MPMATH  ##################


def mndsqrt_mp(mn1, mn2, mn3):
    return mp.matrix(
        [
            [mp.sqrt(mn1), 0, 0],
            [0, mp.sqrt(mn2), 0],
            [0, 0, mp.sqrt(mn3)]
        ]
        )


def MNdsqrt_mp(mn4, mn5, mn6):
    return mp.matrix(
        [
            [mp.sqrt(mn4), 0, 0],
            [0, mp.sqrt(mn5), 0],
            [0, 0, mp.sqrt(mn6)]
        ]
        )
# #############################


def angle(sin2):
    return mp.asin(mp.sqrt(sin2))


th12 = angle(Nudata.sin2theta12.central)
th13 = angle(Nudata.sin2theta13.central)
th23 = angle(Nudata.sin2theta23.central)


Upmns_mp = UpmnsStandardParametrization(
    theta12=th12, theta13=th13, theta23=th23,
    delta=0.0, alpha1=0.0, alpha2=0.0
)


Upmns_mp = mp.matrix([[Upmns_mp[r, s] for r in range(3)] for s in range(3)]).T


# Upmns_mp = mp.matrix([
# [ 0.821302075974486,  0.550502406897554, 0.149699699398496],
# [-0.463050759961518,  0.489988544456971, 0.738576482160108], # 21 corregido
# [ 0.333236993293153, -0.675912957636513, 0.657339166640784]])#Is real


##########################################
def MD_mp(mn1, mn2, mn3, MR1, MR2, MR3, mu1, mu2, mu3):
    return 1j*Upmns_mp*MNdsqrt_mp(
        MR1**2/mu1, MR2**2/mu2, MR3**2/mu3
        )*mndsqrt_mp(mn1, mn2, mn3)


# ###### Diagonalizando con mpmath
def MD_NO(m1, m2, m3, MR1, MR2, MR3, mu1, mu2, mu3):
    return MD_mp(m1, m2, m3, MR1, MR2, MR3, mu1, mu2, mu3)


def Ynu(m1, m2, m3, MR1, MR2, MR3, mu1, mu2, mu3,v):
    return MD_NO(m1, m2, m3, MR1, MR2, MR3, mu1, mu2, mu3)*(mp.sqrt('2')/v)


def Mnu(m1, m2, m3, MR1, MR2, MR3, mu1, mu2, mu3):
    M = MD_NO(m1, m2, m3, MR1, MR2, MR3, mu1, mu2, mu3)
    return mp.matrix(
        [
            [0.0, 0.0, 0.0, M[0, 0], M[0, 1], M[0, 2], 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, M[1, 0], M[1, 1], M[1, 2], 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, M[2, 0], M[2, 1], M[2, 2], 0.0, 0.0, 0.0],
            [M[0, 0], M[1, 0], M[2, 0], 0.0, 0.0, 0.0, MR1, 0.0, 0.0],
            [M[0, 1], M[1, 1], M[2, 1], 0.0, 0.0, 0.0, 0.0, MR2, 0.0],
            [M[0, 2], M[1, 2], M[2, 2], 0.0, 0.0, 0.0, 0.0, 0.0, MR3],
            [0.0, 0.0, 0.0, MR1, 0.0, 0.0, mu1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, MR2, 0.0, 0.0, mu2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, MR3, 0.0, 0.0, mu3]
        ]
    )


def diagonalizationMnu_ISS(m1, m2, m3, MR1, MR2, MR3, mu1, mu2, mu3):
    Mi, UL, UR = mp.eig(
        Mnu(m1, m2, m3, MR1, MR2, MR3, mu1, mu2, mu3),
        left=True, right=True
        )
    Mi, UL, UR = mp.eig_sort(Mi, UL, UR)
    return Mi, UL, UR


def diagonalizationMnu_ISS_svd(m1, m2, m3, MR1, MR2, MR3, mu1, mu2, mu3):
    UL, Mi, UR = mp.svd_c(Mnu(m1, m2, m3, MR1, MR2, MR3, mu1, mu2, mu3))
    return Mi, UL, UR

if __name__ == '__main__':
    v = mp.mpf('246')
    def v1(tb):
        return v*mp.cos(mp.atan(tb))
    
    def v2(tb):
        return v*mp.sin(mp.atan(tb))
    
    def cond_perturvativity(yij):
        return np.abs(yij)**2 < 6*np.pi
        
    def cond_perturvativity_mD(mDij,v):
        return np.abs(mDij)**2 < 12*np.pi/v
    
    #from OneLoopLFVHD.neutrinos import NuOscObservables
    #Nudata = NuOscObservables

    m1 = mp.mpf('1e-12')  #GeV 

    #current values to Square mass differences
    d21 = mp.mpf(str(Nudata.squareDm21.central))*mp.mpf('1e-18')# factor to convert eV^2 to GeV^2
    d31 = mp.mpf(str(Nudata.squareDm31.central))*mp.mpf('1e-18')

    #d21 = 7.5e-5*1e-18
    #d31 = 2.457e-3*1e-18
    m2 = mp.sqrt(m1**2 + d21)
    m3 = mp.sqrt(m1**2 + d31)
    
    M = mp.mpf('1e5')
    mux = mp.mpf('1e-7')
    tb = mp.mpf('1')
    v = mp.mpf('246')

    Y1 = Ynu(m1, m2, m3, M, M, M, mux, mux, mux, v1(tb))
    Y2 = Ynu(m1, m2, m3, M, M, M, mux, mux, mux, v2(tb))

    def conditions_perturvativity_123(Y):
        cond23 = cond_perturvativity(Y[1,2])
        cond13 = cond_perturvativity(Y[0,2])
        cond12 = cond_perturvativity(Y[0,1])
        return (cond23 & cond13 & cond12)
    
    #print(conditions_perturvativity_123(Y1))
    
    
    mD1 = MD_mp(m1, m2, m3, M, M, M, mux, mux, mux)
    
    def conditions_perturvativity_123_mD(mD,v):
        cond23 = cond_perturvativity_mD(mD[1,2], v)
        cond13 = cond_perturvativity_mD(mD[0,2], v)
        cond12 = cond_perturvativity_mD(mD[0,1], v)
        return (cond23 & cond13 & cond12)
    
    #print(conditions_perturvativity_123_mD(mD1,v1(10)))
    #print(conditions_perturvativity_123_mD(mD1,v2(10)))
    
    import pandas as pd
    def df_parameterspace_perturvativity(n):
        tbmp = mp.linspace(-2,3,n)
        tbs = np.array([mp.mpf('10.0')**k for k in tbmp])
        Mmp = mp.linspace(2, 5,n)
        M = np.array([mp.mpf('10.0')**k for k in Mmp])
        muxmp = mp.linspace(-8,-1,n)
        mux = np.array([mp.mpf('10.0')**k for k in muxmp])
    
        tbs_true = []
        M_true = []
        mux_true = []

        tbs_false = []
        M_false = []
        mux_false = []

        for m in M:
            for mu in mux:
                for t in tbs:
                    if conditions_perturvativity_123(Ynu(m1, m2, m3, m, m, m, mu, mu, mu, v1(t))):
                        tbs_true.append(t)
                        M_true.append(m)
                        mux_true.append(mu)
                    else:
                        tbs_false.append(t)
                        M_false.append(m)
                        mux_false.append(mu)
                        
        ps_true1 = pd.DataFrame({'tb':tbs_true, 'M': M_true, 'mux':mux_true})
        ps_false1 = pd.DataFrame({'tb':tbs_false, 'M': M_false, 'mux':mux_false})
        
        tbs_true1 = []
        M_true1 = []
        mux_true1 = []

        tbs_false1 = []
        M_false1 = []
        mux_false1 = []
        
        for m in M:
            for mu in mux:
                for t in tbs:
                    if conditions_perturvativity_123(Ynu(m1, m2, m3, m, m, m, mu, mu, mu, v2(t))):
                        tbs_true1.append(t)
                        M_true1.append(m)
                        mux_true1.append(mu)
                    else:
                        tbs_false1.append(t)
                        M_false1.append(m)
                        mux_false1.append(mu)

        ps_true2 = pd.DataFrame({'tb':tbs_true1, 'M': M_true1, 'mux':mux_true1})
        ps_false2 = pd.DataFrame({'tb':tbs_false1, 'M': M_false1, 'mux':mux_false1})
        
        return ps_true1, ps_true2, ps_false1, ps_false2
    
    
    ps_true1, ps_true2, ps_false1, ps_false2 = df_parameterspace_perturvativity(30)
    
    print(ps_true1)
    print(ps_true2)
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    numbers = [0,ps_true1, ps_true2, ps_false1, ps_false2]
    
    def plot(n):
        data = numbers[n]
        fig = plt.figure()
        plt.scatter(data['M'] , data['mux'], c=data['tb'], cmap='viridis')
        plt.colorbar()
        plt.title(f'Scatter plot {n}')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('$M$')
        plt.xlabel(r'$\mu_X$')
        plt.show()
    
    #plot(1)
    #plot(2)
    #plot(3)
    #plot(4)
    
    def plot_true_false(n1, n2):
        data1 = numbers[n1]
        data2 = numbers[n2]
        #
        fig = plt.figure
        plt.scatter(data1['M'] , data1['mux'], c=data1['tb'], cmap='viridis')
        plt.colorbar()
        plt.title(f'Scatter plot {n1,n2}')
        plt.yscale('log')
        plt.xscale('log')
        #
        plt.scatter(data2['M'] , data2['mux'], alpha=0.1) #, c=data2['tb'], cmap='Spectral')
        # plt.colorbar()
        plt.title(f'Scatter plot {n1,n2}')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('$M$')
        plt.xlabel(r'$\mu_X$')
        plt.show()
   
    #plot_true_false(1, 3)
    #plot_true_false(2, 4)

    def plot3d(n1,n2):
        data1 = numbers[n1]
        data2 = numbers[n2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data1['M'] , data1['mux'], data1['tb'],
            linewidths=1, alpha=.7, edgecolor='k', s = 200, c=data1['tb'])
        
        #ax.scatter(data2['M'] , data2['mux'], data2['tb'],
        #    linewidths=1, alpha=.7, edgecolor='k', s = 200)#, c=data2['tb'])
        
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        # ax.set_zscale('log')
        
        ax.set_xlabel('$M$')
        ax.set_ylabel(r'$\mu_X$')
        ax.set_zlabel(r'$\tan(\beta)$')
        
        plt.show()
        
    plot3d(1, 3)
    plot3d(2, 4)
       



