### Neutrino benchmark
from OneLoopLFVHD.neutrinos import NuOscObservables

Nudata = NuOscObservables

from mpmath import *

mp.dps = 128; mp.pretty = True


m1 = mpf('1e-12')  #GeV 

#current values to Square mass differences
d21 = mpf(str(Nudata.squareDm21.central))*mpf('1e-18')# factor to convert eV^2 to GeV^2
d31 = mpf(str(Nudata.squareDm31.central))*mpf('1e-18')

#d21 = 7.5e-5*1e-18
#d31 = 2.457e-3*1e-18
m2 = sqrt(m1**2 + d21)
m3 = sqrt(m1**2 + d31)

### Numeric translation 
from OneLoopLFVHD.data import ml
# print(ml)

###### 
# Caso degenerado
######
### Diagonalización
n_nu = 9

if n_nu == 9:
    from Unu_seesaw_ISS import diagonalizationMnu_ISS_svd as diagonalizationMnu
    
    diagonalizationMnu1 = lambda m1, m6, mux: diagonalizationMnu(
    m1, m2, m3, m6, m6, m6, mux, mux, mux)
    
    MR = mpf('1e3')
    mux = mpf('1e-7')

    mn,UnuL,UnuR = diagonalizationMnu1(m1, MR, mux)


if n_nu == 6:
    from Unu_seesaw import diagonalizationMnu# as diagonalizationMnu
    
    diagonalizationMnu1 = lambda m1, m6: diagonalizationMnu(
        m1, m2, m3, m6, m6, m6
        )
    
    MR = mpf('1e5')
    mn,UnuL,UnuR = diagonalizationMnu1(m1, MR)
    nprint(chop(mn))


mHpm = mpf('800')
mW = mpf('80.379')
mh = mpf('125.1')

#print('Neutrino masses')
#nprint(chop(mn))
#print('\n')

# a = 2
# b = 3
Unu = UnuL
Unu_dagger = UnuR

Cij = lambda i,j: mp.fsum([UnuL[c,i]*UnuR[j,c] for c in range(3)])

C = matrix(
    [[Cij(i,j) for i in range(n_nu)] for j in range(n_nu)]
)

import OneLoopLFVHD.LFVHDFeynG_mpmath2 as lfvhd_mp

##################################
######## Wninj
##################################

# B012 matrix ninj 

B012_nn = {}
for i in range(n_nu):
    for j in range(n_nu):
        #if i != j:
        B012_nn[(i,j)] = lfvhd_mp.B12_0(mh, mn[i],mn[j])
        #nprint(chop(B012_nn[(i,j)]))

#print(B012_nn)
#B012_nn = matrix(
#    [[
#        if i != j lfvhd_mp.B12_0(mh, mn[i-1],mn[j-1]) else 0 for i in range(9)
#        ] for j in range(9)]
#    )

#print(B012_nn)

def B012_mni2_Cij_sum(a,b):
    return fsum(
    [
        B012_nn[(i-1 ,j-1)]*mn[i-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

B012_mni2_Cij_sum_23 = B012_mni2_Cij_sum(2,3)
B012_mni2_Cij_sum_13 = B012_mni2_Cij_sum(1,3)
B012_mni2_Cij_sum_12 = B012_mni2_Cij_sum(1,2)


def B012_mnj2_Cij_sum(a,b):
    return fsum(
    [
        B012_nn[(i-1 ,j-1)]*mn[j-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

B012_mnj2_Cij_sum_23 = B012_mnj2_Cij_sum(2, 3)
B012_mnj2_Cij_sum_13 = B012_mnj2_Cij_sum(1, 3)
B012_mnj2_Cij_sum_12 = B012_mnj2_Cij_sum(1, 2)

def B012_mnimnj_Cijc_sum(a,b):
    return fsum(
    [
        B012_nn[(i-1 ,j-1)]*mn[i-1]*mn[j-1]*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

B012_mnimnj_Cijc_sum_23 = B012_mnimnj_Cijc_sum(2,3)
B012_mnimnj_Cijc_sum_13 = B012_mnimnj_Cijc_sum(1,3)
B012_mnimnj_Cijc_sum_12 = B012_mnimnj_Cijc_sum(1,2)

# C0 matrix Wninj
C0_Wnn = matrix([
    [
        lfvhd_mp.C0(mh,mW,mn[i],mn[j]) for i in range(n_nu)
        ] for j in range(n_nu)
        ]
    )

# C0 matrix Hninj
C0_Hnn = matrix([
    [
        lfvhd_mp.C0(mh, mHpm, mn[i], mn[j]) for i in range(n_nu)
        ] for j in range(n_nu)
        ]
    )

# C0(Wninj) mni^2
def C0_mni2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C0_Wnn[i-1,j-1]*mn[i-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

C0_mni2Wnn_Cij_sum_23 = C0_mni2Wnn_Cij_sum(2, 3)
C0_mni2Wnn_Cij_sum_13 = C0_mni2Wnn_Cij_sum(1, 3)
C0_mni2Wnn_Cij_sum_12 = C0_mni2Wnn_Cij_sum(1, 2)

# C0(Hninj) mni^2
def C0_mni2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C0_Hnn[i-1,j-1]*mn[i-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

C0_mni2Hnn_Cij_sum_23 = C0_mni2Hnn_Cij_sum(2, 3)
C0_mni2Hnn_Cij_sum_13 = C0_mni2Hnn_Cij_sum(1, 3)
C0_mni2Hnn_Cij_sum_12 = C0_mni2Hnn_Cij_sum(1, 2)

# C0(Wninj) mnj^2
def C0_mnj2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C0_Wnn[i-1,j-1]*mn[j-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

C0_mnj2Wnn_Cij_sum_23 = C0_mnj2Wnn_Cij_sum(2, 3)
C0_mnj2Wnn_Cij_sum_13 = C0_mnj2Wnn_Cij_sum(1, 3)
C0_mnj2Wnn_Cij_sum_12 = C0_mnj2Wnn_Cij_sum(1, 2)

# C0(Hninj) mnj^2
def C0_mnj2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C0_Hnn[i-1,j-1]*mn[j-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

C0_mnj2Hnn_Cij_sum_23 = C0_mnj2Hnn_Cij_sum(2, 3)
C0_mnj2Hnn_Cij_sum_13 = C0_mnj2Hnn_Cij_sum(1, 3)
C0_mnj2Hnn_Cij_sum_12 = C0_mnj2Hnn_Cij_sum(1, 2)

# C0(Wninj) mni mnj Cijc
def C0_mnimnjWnn_Cijc_sum(a, b):
    return fsum(
    [
        C0_Wnn[i-1,j-1]*mn[i-1]*mn[j-1]*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

C0_mnimnjWnn_Cijc_sum_23 = C0_mnimnjWnn_Cijc_sum(2, 3)
C0_mnimnjWnn_Cijc_sum_13 = C0_mnimnjWnn_Cijc_sum(1, 3)
C0_mnimnjWnn_Cijc_sum_12 = C0_mnimnjWnn_Cijc_sum(1, 2)

# C0(Hninj) mni mnj Cijc
def C0_mnimnjHnn_Cijc_sum(a, b):
    return fsum(
    [
        C0_Hnn[i-1,j-1]*mn[i-1]*mn[j-1]*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

C0_mnimnjHnn_Cijc_sum_23 = C0_mnimnjHnn_Cijc_sum(2, 3)
C0_mnimnjHnn_Cijc_sum_13 = C0_mnimnjHnn_Cijc_sum(1, 3)
C0_mnimnjHnn_Cijc_sum_12 = C0_mnimnjHnn_Cijc_sum(1, 2)

# C0(Hninj) mni^2 mnj^2
def C0_mni2mnj2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C0_Hnn[i-1,j-1]*mn[i-1]**2*mn[j-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

C0_mni2mnj2Hnn_Cij_sum_23 = C0_mni2mnj2Hnn_Cij_sum(2, 3)
C0_mni2mnj2Hnn_Cij_sum_13 = C0_mni2mnj2Hnn_Cij_sum(1, 3)
C0_mni2mnj2Hnn_Cij_sum_12 = C0_mni2mnj2Hnn_Cij_sum(1, 2)


# C0(Hninj) mni mnj^3 Cijc
def C0_mnimnj3Hnn_Cijc_sum(a, b):
    return fsum(
    [
        C0_Hnn[i-1,j-1]*mn[i-1]*mn[j-1]**3*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

C0_mnimnj3Hnn_Cijc_sum_23 = C0_mnimnj3Hnn_Cijc_sum(2, 3)
C0_mnimnj3Hnn_Cijc_sum_13 = C0_mnimnj3Hnn_Cijc_sum(1, 3)
C0_mnimnj3Hnn_Cijc_sum_12 = C0_mnimnj3Hnn_Cijc_sum(1, 2)

# C0(Hninj) mni^3 mnj Cijc 
def C0_mni3mnjHnn_Cijc_sum(a, b):
    return fsum(
    [
        C0_Hnn[i-1,j-1]*mn[i-1]**3*mn[j-1]*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

C0_mni3mnjHnn_Cijc_sum_23 = C0_mni3mnjHnn_Cijc_sum(2, 3)
C0_mni3mnjHnn_Cijc_sum_13 = C0_mni3mnjHnn_Cijc_sum(1, 3)
C0_mni3mnjHnn_Cijc_sum_12 = C0_mni3mnjHnn_Cijc_sum(1, 2)

###########################################
############### C1
###########################################
# C1 matrix Wninj
def C1_Wnn(a):
    return matrix([
        [
            lfvhd_mp.C1(mh, ml[a], mW, mn[i], mn[j]) for i in range(n_nu)
            ] for j in range(n_nu)
            ]
        )

C1_Wnn_2 = C1_Wnn(2)
C1_Wnn_1 = C1_Wnn(1)

def C1_Wnn_caso(a):
    if a == 1:
        C1_Wnn_a = C1_Wnn_1
    elif a == 2:
        C1_Wnn_a = C1_Wnn_2
    else:
        raise ValueError('a must be equals to 1 or 2')
    
    return C1_Wnn_a

# C1 matrix Hninj
def C1_Hnn(a):
    return matrix([
        [
            lfvhd_mp.C1(mh, ml[a], mHpm, mn[i], mn[j]) for i in range(n_nu)
            ] for j in range(n_nu)
        ]
        )

C1_Hnn_2 = C1_Hnn(2)
C1_Hnn_1 = C1_Hnn(1) 

def C1_Hnn_caso(a):
    if a == 1:
        C1_Hnn_a = C1_Wnn_1
    elif a == 2:
        C1_Hnn_a = C1_Wnn_2
    else:
        raise ValueError('a must be equals to 1 or 2')
    
    return C1_Hnn_a

# C1(Wninj)mnj^2
def C1_mnj2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C1_Wnn_caso(a)[i-1,j-1]*mn[j-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )
C1_mnj2Wnn_Cij_sum_23 = C1_mnj2Wnn_Cij_sum(2, 3)
C1_mnj2Wnn_Cij_sum_13 = C1_mnj2Wnn_Cij_sum(1, 3)
C1_mnj2Wnn_Cij_sum_12 = C1_mnj2Wnn_Cij_sum(1, 2)

# C1(Hninj)mnj^2
def C1_mnj2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C1_Hnn_caso(a)[i-1,j-1]*mn[j-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

C1_mnj2Hnn_Cij_sum_23 = C1_mnj2Hnn_Cij_sum(2, 3)
C1_mnj2Hnn_Cij_sum_13 = C1_mnj2Hnn_Cij_sum(1, 3)
C1_mnj2Hnn_Cij_sum_12 = C1_mnj2Hnn_Cij_sum(1, 2)

# C1(Wninj)mni^2
def C1_mni2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C1_Wnn_caso(a)[i-1,j-1]*mn[i-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

C1_mni2Wnn_Cij_sum_23 = C1_mni2Wnn_Cij_sum(2, 3)
C1_mni2Wnn_Cij_sum_13 = C1_mni2Wnn_Cij_sum(1, 3)
C1_mni2Wnn_Cij_sum_12 = C1_mni2Wnn_Cij_sum(1, 2)

# C1(Hninj)mni^2
def C1_mni2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C1_Hnn_caso(a)[i-1,j-1]*mn[i-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

C1_mni2Hnn_Cij_sum_23 = C1_mni2Hnn_Cij_sum(2, 3)
C1_mni2Hnn_Cij_sum_13 = C1_mni2Hnn_Cij_sum(1, 3)
C1_mni2Hnn_Cij_sum_12 = C1_mni2Hnn_Cij_sum(1, 2)

# C1(Wninj)mni^2 mnj^2
def C1_mni2mnj2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C1_Wnn_caso(a)[i-1,j-1]*mn[i-1]**2*mn[j-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)
        for j in range(1, n_nu + 1) # if i != j
        ]
    )

C1_mni2mnj2Wnn_Cij_sum_23 = C1_mni2mnj2Wnn_Cij_sum(2,3)
C1_mni2mnj2Wnn_Cij_sum_13 = C1_mni2mnj2Wnn_Cij_sum(1,3)
C1_mni2mnj2Wnn_Cij_sum_12 = C1_mni2mnj2Wnn_Cij_sum(1,2)

# C1(Hninj)mni^2 mnj^2
def C1_mni2mnj2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C1_Hnn_caso(a)[i-1,j-1]*mn[i-1]**2*mn[j-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)
        for j in range(1, n_nu + 1) # if i != j
        ]
    )

C1_mni2mnj2Hnn_Cij_sum_23 = C1_mni2mnj2Hnn_Cij_sum(2, 3)
C1_mni2mnj2Hnn_Cij_sum_13 = C1_mni2mnj2Hnn_Cij_sum(1, 3)
C1_mni2mnj2Hnn_Cij_sum_12 = C1_mni2mnj2Hnn_Cij_sum(1, 2)

# C1(Wninj)mni mnj Cijc
def C1_mnimnjWnn_Cijc_sum(a, b):
    return fsum(
    [
        C1_Wnn_caso(a)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)
        for j in range(1, n_nu + 1) # if i != j
        ]
    )

C1_mnimnjWnn_Cijc_sum_23 = C1_mnimnjWnn_Cijc_sum(2, 3)
C1_mnimnjWnn_Cijc_sum_13 = C1_mnimnjWnn_Cijc_sum(1, 3)
C1_mnimnjWnn_Cijc_sum_12 = C1_mnimnjWnn_Cijc_sum(1, 2)

# C1(Hninj)mni mnj Cijc
def C1_mnimnjHnn_Cijc_sum(a, b):
    return fsum(
    [
        C1_Hnn_caso(a)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)
        for j in range(1, n_nu + 1) # if i != j
        ]
    )

C1_mnimnjHnn_Cijc_sum_23 = C1_mnimnjHnn_Cijc_sum(2, 3)
C1_mnimnjHnn_Cijc_sum_13 = C1_mnimnjHnn_Cijc_sum(1, 3)
C1_mnimnjHnn_Cijc_sum_12 = C1_mnimnjHnn_Cijc_sum(1, 2)

# (mni^2 + mnj^2) C1(Wninj)
def mni2_mnj2_C1_Wnn_Cijc_sum(a, b):
    return fsum(
    [
        (mn[i-1]**2 + mn[j-1]**2)*C1_Wnn_caso(a)[i-1,j-1]*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)
        for j in range(1, n_nu + 1) # if i != j
        ]
    )

mni2_mnj2_C1_Wnn_Cijc_sum_23 = mni2_mnj2_C1_Wnn_Cijc_sum(2, 3)
mni2_mnj2_C1_Wnn_Cijc_sum_13 = mni2_mnj2_C1_Wnn_Cijc_sum(1, 3)
mni2_mnj2_C1_Wnn_Cijc_sum_12 = mni2_mnj2_C1_Wnn_Cijc_sum(1, 2)

# (mni^2 + mnj^2) C1(Wninj) mni mnj Cijc
def mni2_mnj2_C1_mnimnjWnn_Cijc_sum(a, b):
    return fsum(
    [
        (mn[i-1]**2 + mn[j-1]**2)*C1_Wnn_caso(a)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

mni2_mnj2_C1_mnimnjWnn_Cijc_sum_23 = mni2_mnj2_C1_mnimnjWnn_Cijc_sum(2, 3)
mni2_mnj2_C1_mnimnjWnn_Cijc_sum_13 = mni2_mnj2_C1_mnimnjWnn_Cijc_sum(1, 3)
mni2_mnj2_C1_mnimnjWnn_Cijc_sum_12 = mni2_mnj2_C1_mnimnjWnn_Cijc_sum(1, 2)

# (mni^2 + mnj^2) C1(Hninj) mni mnj Cijc
def mni2_mnj2_C1_mnimnjHnn_Cijc_sum(a, b):
    return fsum(
    [
        (mn[i-1]**2 + mn[j-1]**2)*C1_Hnn_caso(a)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

mni2_mnj2_C1_mnimnjHnn_Cijc_sum_23 = mni2_mnj2_C1_mnimnjHnn_Cijc_sum(2, 3)
mni2_mnj2_C1_mnimnjHnn_Cijc_sum_13 = mni2_mnj2_C1_mnimnjHnn_Cijc_sum(1, 3)
mni2_mnj2_C1_mnimnjHnn_Cijc_sum_12 = mni2_mnj2_C1_mnimnjHnn_Cijc_sum(1, 2)

###########################################
############### C2
###########################################
# C2 matrix Wninj
def C2_Wnn(b):
    return matrix([
    [
        lfvhd_mp.C2(mh, ml[b], mW, mn[i], mn[j]) for i in range(n_nu)
        ] for j in range(n_nu)
        ]
)

C2_Wnn_2 = C2_Wnn(2)
C2_Wnn_3 = C2_Wnn(3)

def C2_Wnn_caso(b):
    if b == 2:
        C2_Wnn_b = C2_Wnn_2
    elif b == 3:
        C2_Wnn_b = C2_Wnn_3
    else:
        raise ValueError('b must be equals to 2 or 3')
    
    return C2_Wnn_b

# C2 matrix Hninj
def C2_Hnn(b):
    return matrix([
    [
        lfvhd_mp.C2(mh, ml[b], mHpm, mn[i], mn[j]) for i in range(n_nu)
        ] for j in range(n_nu)
        ]
)

C2_Hnn_2 = C2_Hnn(2)
C2_Hnn_3 = C2_Hnn(3)

def C2_Hnn_caso(b):
    if b == 2:
        C2_Hnn_b = C2_Hnn_2
    elif b == 3:
        C2_Hnn_b = C2_Hnn_3
    else:
        raise ValueError('b must be equals to 2 or 3')
    
    return C2_Hnn_b

# C2(Wninj)*mnj^2
def C2_mnj2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C2_Wnn_caso(b)[i-1,j-1]*mn[j-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

C2_mnj2Wnn_Cij_sum_23 = C2_mnj2Wnn_Cij_sum(2, 3)
C2_mnj2Wnn_Cij_sum_13 = C2_mnj2Wnn_Cij_sum(1, 3)
C2_mnj2Wnn_Cij_sum_12 = C2_mnj2Wnn_Cij_sum(1, 2)

# C2(Hninj)*mnj^2
def C2_mnj2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C2_Hnn_caso(b)[i-1,j-1]*mn[j-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

C2_mnj2Hnn_Cij_sum_23 = C2_mnj2Hnn_Cij_sum(2, 3)
C2_mnj2Hnn_Cij_sum_13 = C2_mnj2Hnn_Cij_sum(1, 3)
C2_mnj2Hnn_Cij_sum_12 = C2_mnj2Hnn_Cij_sum(1, 2)

# C2(Wninj)*mni^2
def C2_mni2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C2_Wnn_caso(b)[i-1,j-1]*mn[i-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

C2_mni2Wnn_Cij_sum_23 = C2_mni2Wnn_Cij_sum(2, 3)
C2_mni2Wnn_Cij_sum_13 = C2_mni2Wnn_Cij_sum(1, 3)
C2_mni2Wnn_Cij_sum_12 = C2_mni2Wnn_Cij_sum(1, 2)

# C2(Hninj)*mni^2
def C2_mni2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C2_Hnn_caso(b)[i-1,j-1]*mn[i-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1) for j in range(1, n_nu + 1)
        # if i != j
        ]
    )

C2_mni2Hnn_Cij_sum_23 = C2_mni2Hnn_Cij_sum(2, 3)
C2_mni2Hnn_Cij_sum_13 = C2_mni2Hnn_Cij_sum(1, 3)
C2_mni2Hnn_Cij_sum_12 = C2_mni2Hnn_Cij_sum(1, 2)

# C2(Wninj)*mni^2*mnj^2
def C2_mni2mnj2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C2_Wnn_caso(b)[i-1,j-1]*mn[i-1]**2*mn[j-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) #if i != j
        ]
    )

C2_mni2mnj2Wnn_Cij_sum_23 = C2_mni2mnj2Wnn_Cij_sum(2, 3)
C2_mni2mnj2Wnn_Cij_sum_13 = C2_mni2mnj2Wnn_Cij_sum(1, 3)
C2_mni2mnj2Wnn_Cij_sum_12 = C2_mni2mnj2Wnn_Cij_sum(1, 2)

# C2(Hninj)*mni^2*mnj^2
def C2_mni2mnj2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C2_Hnn_caso(b)[i-1,j-1]*mn[i-1]**2*mn[j-1]**2*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

C2_mni2mnj2Hnn_Cij_sum_23 = C2_mni2mnj2Hnn_Cij_sum(2, 3)
C2_mni2mnj2Hnn_Cij_sum_13 = C2_mni2mnj2Hnn_Cij_sum(1, 3)
C2_mni2mnj2Hnn_Cij_sum_12 = C2_mni2mnj2Hnn_Cij_sum(1, 2)

# C2(Wninj)*mni*mnj Cijc
def C2_mnimnjWnn_Cijc_sum(a, b):
    return fsum(
    [
        C2_Wnn_caso(b)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

C2_mnimnjWnn_Cijc_sum_23 = C2_mnimnjWnn_Cijc_sum(2, 3)
C2_mnimnjWnn_Cijc_sum_13 = C2_mnimnjWnn_Cijc_sum(1, 3)
C2_mnimnjWnn_Cijc_sum_12 = C2_mnimnjWnn_Cijc_sum(1, 2)

# C2(Hninj)*mni*mnj Cijc
def C2_mnimnjHnn_Cijc_sum(a, b):
    return fsum(
    [
        C2_Hnn_caso(b)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

C2_mnimnjHnn_Cijc_sum_23 = C2_mnimnjHnn_Cijc_sum(2, 3)
C2_mnimnjHnn_Cijc_sum_13 = C2_mnimnjHnn_Cijc_sum(1, 3)
C2_mnimnjHnn_Cijc_sum_12 = C2_mnimnjHnn_Cijc_sum(1, 2)

# (mni^2 + mnj^2) C2(Wninj)
def mni2_mnj2_C2_Wnn_Cijc_sum(a, b):
    return fsum(
    [
        (mn[i-1]**2 + mn[j-1]**2)*C2_Wnn_caso(b)[i-1,j-1]*C[i-1, j-1]*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

mni2_mnj2_C2_Wnn_Cijc_sum_23 = mni2_mnj2_C2_Wnn_Cijc_sum(2, 3)
mni2_mnj2_C2_Wnn_Cijc_sum_13 = mni2_mnj2_C2_Wnn_Cijc_sum(1, 3)
mni2_mnj2_C2_Wnn_Cijc_sum_12 = mni2_mnj2_C2_Wnn_Cijc_sum(1, 2)


# (mni^2 + mnj^2) C2(Wninj)*mni*mnj
def mni2_mnj2_C2_mnimnjWnn_Cijc_sum(a, b):
    return fsum(
    [
        (mn[i-1]**2 + mn[j-1]**2)*C2_Wnn_caso(b)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

mni2_mnj2_C2_mnimnjWnn_Cijc_sum_23 = mni2_mnj2_C2_mnimnjWnn_Cijc_sum(2, 3)
mni2_mnj2_C2_mnimnjWnn_Cijc_sum_13 = mni2_mnj2_C2_mnimnjWnn_Cijc_sum(1, 3)
mni2_mnj2_C2_mnimnjWnn_Cijc_sum_12 = mni2_mnj2_C2_mnimnjWnn_Cijc_sum(1, 2)

# (mni^2 + mnj^2) C2(Hninj)*mni*mnj
def mni2_mnj2_C2_mnimnjHnn_Cijc_sum(a, b):
    return fsum(
    [
        (mn[i-1]**2 + mn[j-1]**2)*C2_Hnn_caso(b)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(C[i-1, j-1])*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, n_nu + 1) for j in range(1, n_nu + 1) # if i != j
        ]
    )

mni2_mnj2_C2_mnimnjHnn_Cijc_sum_23 = mni2_mnj2_C2_mnimnjHnn_Cijc_sum(2, 3)
mni2_mnj2_C2_mnimnjHnn_Cijc_sum_13 = mni2_mnj2_C2_mnimnjHnn_Cijc_sum(1, 3)
mni2_mnj2_C2_mnimnjHnn_Cijc_sum_12 = mni2_mnj2_C2_mnimnjHnn_Cijc_sum(1, 2)


##################################
######## PV functions one neutrino
##################################

# B10 matrix niW
B10_nW = lambda a: [
    lfvhd_mp.B1_0(ml[a],mn[i],mW) for i in range(n_nu)
    ]

B10_nW_2 = B10_nW(2)
B10_nW_1 = B10_nW(1)

def B10_nW_caso(a):
    if a == 1:
        B10_nW_a = B10_nW_1
    elif a == 2:
        B10_nW_a = B10_nW_2
    else:
        raise ValueError('a must be equals to 1 or 2')
    
    return B10_nW_a

# B10 matrix niH
B10_nH = lambda a: [
    lfvhd_mp.B1_0(ml[a],mn[i],mHpm) for i in range(n_nu)
    ]

B10_nH_2 = B10_nH(2)
B10_nH_1 = B10_nH(1)

def B10_nH_caso(a):
    if a == 1:
        B10_nH_a = B10_nH_1
    elif a == 2:
        B10_nH_a = B10_nH_2
    else:
        raise ValueError('a must be equals to 1 or 2')
    
    return B10_nH_a

# B10_nW mni^2 
def B10_nW_mni2_UU_sum(a,b):
    return fsum(
        [B10_nW_caso(a)[i-1]*mn[i-1]**2*Unu[b-1,i-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)]
        )

B10_nW_mni2_UU_sum_23 = B10_nW_mni2_UU_sum(2, 3)
B10_nW_mni2_UU_sum_13 = B10_nW_mni2_UU_sum(1, 3)
B10_nW_mni2_UU_sum_12 = B10_nW_mni2_UU_sum(1, 2)

# B10_nH mni^2 
def B10_nH_mni2_UU_sum(a,b):
    return fsum(
        [B10_nH_caso(a)[i-1]*mn[i-1]**2*Unu[b-1,i-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)]
        )

B10_nH_mni2_UU_sum_23 = B10_nH_mni2_UU_sum(2, 3)
B10_nH_mni2_UU_sum_13 = B10_nH_mni2_UU_sum(1, 3)
B10_nH_mni2_UU_sum_12 = B10_nH_mni2_UU_sum(1, 2)

# B20 matrix niW
B20_nW = lambda b: [
    lfvhd_mp.B1_0(ml[b],mn[i],mW) for i in range(n_nu)
    ]

B20_nW_3 = B20_nW(3)
B20_nW_2 = B20_nW(2)

def B20_nW_caso(b):
    if b == 2:
        B20_nW_b = B20_nW_2
    elif b == 3:
        B20_nW_b = B20_nW_3
    else:
        raise ValueError('b must be equals to 2 or 3')
    
    return B20_nW_b

# B20 mni^2 
def B20_nW_mni2_UU_sum(a,b):
    return fsum(
        [B20_nW_caso(b)[i-1]*mn[i-1]**2*Unu[b-1,i-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)]
        )

B20_nW_mni2_UU_sum_23 = B20_nW_mni2_UU_sum(2, 3)
B20_nW_mni2_UU_sum_13 = B20_nW_mni2_UU_sum(1, 3)
B20_nW_mni2_UU_sum_12 = B20_nW_mni2_UU_sum(1, 2)

# B20 matrix niH
B20_nH = lambda b: [
    lfvhd_mp.B1_0(ml[b],mn[i],mHpm) for i in range(n_nu)
    ]

B20_nH_3 = B20_nH(3)
B20_nH_2 = B20_nH(2)

def B20_nH_caso(b):
    if b == 2:
        B20_nH_b = B20_nH_2
    elif b == 3:
        B20_nH_b = B20_nH_3
    else:
        raise ValueError('b must be equals to 2 or 3')
    
    return B20_nH_b

# B20 mni^2 
def B20_nH_mni2_UU_sum(a,b):
    return fsum(
        [B20_nH_caso(b)[i-1]*mn[i-1]**2*Unu[b-1,i-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)]
        )

B20_nH_mni2_UU_sum_23 = B20_nH_mni2_UU_sum(2, 3)
B20_nH_mni2_UU_sum_13 = B20_nH_mni2_UU_sum(1, 3)
B20_nH_mni2_UU_sum_12 = B20_nH_mni2_UU_sum(1, 2)

# B11 matrix niW
B11_nW = lambda a: [
    lfvhd_mp.B1_1(ml[a],mn[i],mW) for i in range(n_nu)
    ]


B11_nW_2 = B11_nW(2)
B11_nW_1 = B11_nW(1)

def B11_nW_caso(a):
    if a == 1:
        B11_nW_a = B11_nW_1
    elif a == 2:
        B11_nW_a = B11_nW_2
    else:
        raise ValueError('a must be equals to 1 or 2')
    
    return B11_nW_a

# B11 matrix niH
B11_nH = lambda a: [
    lfvhd_mp.B1_1(ml[a],mn[i],mHpm) for i in range(n_nu)
    ]


B11_nH_2 = B11_nH(2)
B11_nH_1 = B11_nH(1)

def B11_nH_caso(a):
    if a == 1:
        B11_nH_a = B11_nH_1
    elif a == 2:
        B11_nH_a = B11_nH_2
    else:
        raise ValueError('a must be equals to 1 or 2')
    
    return B11_nH_a

# B11_nW
def B11_nW_UU_sum(a,b):
    return fsum(
        [B11_nW_caso(a)[i-1]*Unu[b-1,i-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)]
    )

B11_nW_UU_sum_23 = B11_nW_UU_sum(2, 3)
B11_nW_UU_sum_13 = B11_nW_UU_sum(1, 3)
B11_nW_UU_sum_12 = B11_nW_UU_sum(1, 2)

# B11_nH
def B11_nH_UU_sum(a,b):
    return fsum(
        [B11_nH_caso(a)[i-1]*Unu[b-1,i-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)]
    )

B11_nH_UU_sum_23 = B11_nH_UU_sum(2, 3)
B11_nH_UU_sum_13 = B11_nH_UU_sum(1, 3)
B11_nH_UU_sum_12 = B11_nH_UU_sum(1, 2)

# B11_nW mni^2
def B11_nW_mni2_UU_sum(a,b):
    return fsum(
        [B11_nW_caso(a)[i-1]*mn[i-1]**2*Unu[b-1,i-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)]
    )

B11_nW_mni2_UU_sum_23 = B11_nW_mni2_UU_sum(2, 3)
B11_nW_mni2_UU_sum_13 = B11_nW_mni2_UU_sum(1, 3)
B11_nW_mni2_UU_sum_12 = B11_nW_mni2_UU_sum(1, 2)

# B11_nH mni^2
def B11_nH_mni2_UU_sum(a,b):
    return fsum(
        [B11_nH_caso(a)[i-1]*mn[i-1]**2*Unu[b-1,i-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)]
    )

B11_nH_mni2_UU_sum_23 = B11_nH_mni2_UU_sum(2, 3)
B11_nH_mni2_UU_sum_13 = B11_nH_mni2_UU_sum(1, 3)
B11_nH_mni2_UU_sum_12 = B11_nH_mni2_UU_sum(1, 2)


# B21 matrix niW
B21_nW = lambda b: [
    lfvhd_mp.B2_1(ml[b],mn[i],mW) for i in range(n_nu)
    ]


B21_nW_2 = B21_nW(2)
B21_nW_3 = B21_nW(3)

def B21_nW_caso(b):
    if b == 2:
        B21_nW_b = B21_nW_2
    elif b == 3:
        B21_nW_b = B21_nW_3
    else:
        raise ValueError('b must be equals to 2 or 3')
    
    return B21_nW_b

def B21_nW_UU_sum(a,b):
    return fsum(
        [B21_nW_caso(b)[i-1]*Unu[b-1,i-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)]
    )

B21_nW_UU_sum_23 = B21_nW_UU_sum(2, 3)
B21_nW_UU_sum_13 = B21_nW_UU_sum(1, 3)
B21_nW_UU_sum_12 = B21_nW_UU_sum(1, 2)

def B21_nW_mni2_UU_sum(a,b):
    return fsum(
        [B21_nW_caso(b)[i-1]*mn[i-1]**2*Unu[b-1,i-1]*Unu_dagger[i-1,a-1] for i in range(1, n_nu + 1)]
        )

B21_nW_mni2_UU_sum_23 = B21_nW_mni2_UU_sum(2, 3)
B21_nW_mni2_UU_sum_13 = B21_nW_mni2_UU_sum(1, 3)
B21_nW_mni2_UU_sum_12 = B21_nW_mni2_UU_sum(1, 2)

# B21 matrix niH
B21_nH = lambda b: [
    lfvhd_mp.B2_1(ml[b],mn[i],mHpm) for i in range(n_nu)
    ]


B21_nH_2 = B21_nH(2)
B21_nH_3 = B21_nH(3)

def B21_nH_caso(b):
    if b == 2:
        B21_nH_b = B21_nH_2
    elif b == 3:
        B21_nH_b = B21_nH_3
    else:
        raise ValueError('b must be equals to 2 or 3')
    
    return B21_nH_b

def B21_nH_UU_sum(a,b):
    return fsum(
        [B21_nH_caso(b)[i-1]*Unu[b-1,i-1]*Unu_dagger[i-1,a-1] for i in range(1,n_nu + 1)]
    )

B21_nH_UU_sum_23 = B21_nH_UU_sum(2, 3)
B21_nH_UU_sum_13 = B21_nH_UU_sum(1, 3)
B21_nH_UU_sum_12 = B21_nH_UU_sum(1, 2)

def B21_nH_mni2_UU_sum(a,b):
    return fsum(
        [B21_nH_caso(b)[i-1]*mn[i-1]**2*Unu[b-1,i-1]*Unu_dagger[i-1,a-1] for i in range(1,n_nu + 1)]
        )

B21_nH_mni2_UU_sum_23 = B21_nH_mni2_UU_sum(2, 3)
B21_nH_mni2_UU_sum_13 = B21_nH_mni2_UU_sum(1, 3)
B21_nH_mni2_UU_sum_12 = B21_nH_mni2_UU_sum(1, 2)

from modelos_2HDM import coeff_typeI_h as coeff_h
#from modelos_2HDM import coeff_typeI_h as coeff_h
#from modelos_2HDM import coeff_lepton_specific_h as coeff_h
#from modelos_2HDM import coeff_flipped_h as coeff_h

from modelos_2HDM import tb, cab
from sympy import lambdify
#mA = symbols('m_A',positive=True)
#Kphi =  4*mA**2 - 3*mϕ**2- 2*mHpm**2
#Qphi = mϕ**2 - 2*mHpm**2

#print('coeffs = ', coeff_h.xi_lA*coeff_h.xi_nA)

xi_nphi = lambdify([tb, cab], coeff_h.xi_nphi, 'mpmath')
xi_lphi = lambdify([tb, cab], coeff_h.xi_lphi, 'mpmath')
xi_lA = lambdify([tb], coeff_h.xi_lA, 'mpmath')
xi_nA = lambdify([tb], coeff_h.xi_nA, 'mpmath')

v = mpf('246')
g = 2*mW/v
F = g**3/(64*pi**2*mW**3)
tb0 = mpf('1e-2')
cab0 = mpf('1e-2')

from OneLoopLFVHD import Γhlilj

class Higgs_to_ll():
    def __init__(
        self, a, b,
        C1_mnimnjWnn_Cijc_sum_ij_a,
        C2_mnimnjWnn_Cijc_sum_ij_b,
        C2_mni2Hnn_Cij_sum_ij_b,
        C2_mnj2Hnn_Cij_sum_ij_b,
        C1_mnj2Hnn_Cij_sum_ij_a,
        C1_mni2Hnn_Cij_sum_ij_a,
        C2_mnimnjHnn_Cijc_sum_ij_b,
        C1_mnimnjHnn_Cijc_sum_ij_a,
        ################
        C1_mnj2Wnn_Cij_sum_ij_a,
        C1_mni2Wnn_Cij_sum_ij_a,
        C1_mni2mnj2Wnn_Cij_sum_ij_a,
        mni2_mnj2_C1_mnimnjWnn_Cijc_sum_ij_a,
        C2_mnj2Wnn_Cij_sum_ij_b,
        C2_mni2Wnn_Cij_sum_ij_b,
        C2_mni2mnj2Wnn_Cij_sum_ij_b,
        mni2_mnj2_C2_mnimnjWnn_Cijc_sum_ij_b,
        mni2_mnj2_C1_Wnn_Cijc_sum_ij_a,
        mni2_mnj2_C2_Wnn_Cijc_sum_ij_b,
        C1_mni2mnj2Hnn_Cij_sum_ij_a,
        mni2_mnj2_C1_mnimnjHnn_Cijc_sum_ij_a,
        C2_mni2mnj2Hnn_Cij_sum_ij_b,
        mni2_mnj2_C2_mnimnjHnn_Cijc_sum_ij_b,
        ############################ 
        B012_mni2_Cij_sum_ij,
        B012_mnj2_Cij_sum_ij,
        B012_mnimnj_Cijc_sum_ij,
        C0_mni2Wnn_Cij_sum_ij,
        C0_mni2Hnn_Cij_sum_ij,
        C0_mnj2Wnn_Cij_sum_ij,
        C0_mnj2Hnn_Cij_sum_ij,
        C0_mnimnjWnn_Cijc_sum_ij,
        C0_mnimnjHnn_Cijc_sum_ij,
        C0_mni2mnj2Hnn_Cij_sum_ij,
        C0_mnimnj3Hnn_Cijc_sum_ij,
        C0_mni3mnjHnn_Cijc_sum_ij,
        ############# One neutrino##############
        B11_nW_UU_sum_i_a,
        B21_nW_UU_sum_i_b,
        B10_nW_mni2_UU_sum_i_a,
        B20_nW_mni2_UU_sum_i_b,
        B11_nW_mni2_UU_sum_i_a,
        B21_nW_mni2_UU_sum_i_b,
        #######
        B11_nH_UU_sum_i_a,
        B11_nH_mni2_UU_sum_i_a,
        B10_nH_mni2_UU_sum_i_a,
        B20_nH_mni2_UU_sum_i_b,
        B21_nH_UU_sum_i_b,
        B21_nH_mni2_UU_sum_i_b
        ):
        ####################
        ####################
        self.a = a
        self.b = b
        self.C1_mnimnjWnn_Cijc_sum_ij_a = C1_mnimnjWnn_Cijc_sum_ij_a
        self.C2_mnimnjWnn_Cijc_sum_ij_b = C2_mnimnjWnn_Cijc_sum_ij_b
        self.C2_mni2Hnn_Cij_sum_ij_b = C2_mni2Hnn_Cij_sum_ij_b
        self.C2_mnj2Hnn_Cij_sum_ij_b = C2_mnj2Hnn_Cij_sum_ij_b
        self.C1_mnj2Hnn_Cij_sum_ij_a = C1_mnj2Hnn_Cij_sum_ij_a
        self.C1_mni2Hnn_Cij_sum_ij_a = C1_mni2Hnn_Cij_sum_ij_a
        self.C2_mnimnjHnn_Cijc_sum_ij_b = C2_mnimnjHnn_Cijc_sum_ij_b
        self.C1_mnimnjHnn_Cijc_sum_ij_a = C1_mnimnjHnn_Cijc_sum_ij_a
        ################
        self.C1_mnj2Wnn_Cij_sum_ij_a = C1_mnj2Wnn_Cij_sum_ij_a
        self.C1_mni2Wnn_Cij_sum_ij_a = C1_mni2Wnn_Cij_sum_ij_a
        self.C1_mni2mnj2Wnn_Cij_sum_ij_a = C1_mni2mnj2Wnn_Cij_sum_ij_a
        self.mni2_mnj2_C1_mnimnjWnn_Cijc_sum_ij_a = mni2_mnj2_C1_mnimnjWnn_Cijc_sum_ij_a
        self.C2_mnj2Wnn_Cij_sum_ij_b = C2_mnj2Wnn_Cij_sum_ij_b
        self.C2_mni2Wnn_Cij_sum_ij_b = C2_mni2Wnn_Cij_sum_ij_b
        self.C2_mni2mnj2Wnn_Cij_sum_ij_b = C2_mni2mnj2Wnn_Cij_sum_ij_b
        self.mni2_mnj2_C2_mnimnjWnn_Cijc_sum_ij_b = mni2_mnj2_C2_mnimnjWnn_Cijc_sum_ij_b
        self.mni2_mnj2_C1_Wnn_Cijc_sum_ij_a = mni2_mnj2_C1_Wnn_Cijc_sum_ij_a
        self.mni2_mnj2_C2_Wnn_Cijc_sum_ij_b = mni2_mnj2_C2_Wnn_Cijc_sum_ij_b
        self.C1_mni2mnj2Hnn_Cij_sum_ij_a = C1_mni2mnj2Hnn_Cij_sum_ij_a
        self.mni2_mnj2_C1_mnimnjHnn_Cijc_sum_ij_a = mni2_mnj2_C1_mnimnjHnn_Cijc_sum_ij_a
        self.C2_mni2mnj2Hnn_Cij_sum_ij_b = C2_mni2mnj2Hnn_Cij_sum_ij_b
        self.mni2_mnj2_C2_mnimnjHnn_Cijc_sum_ij_b = mni2_mnj2_C2_mnimnjHnn_Cijc_sum_ij_b
        ############################ 
        self.B012_mni2_Cij_sum_ij = B012_mni2_Cij_sum_ij
        self.B012_mnj2_Cij_sum_ij = B012_mnj2_Cij_sum_ij
        self.B012_mnimnj_Cijc_sum_ij = B012_mnimnj_Cijc_sum_ij
        self.C0_mni2Wnn_Cij_sum_ij = C0_mni2Wnn_Cij_sum_ij
        self.C0_mni2Hnn_Cij_sum_ij = C0_mni2Hnn_Cij_sum_ij
        self.C0_mnj2Wnn_Cij_sum_ij = C0_mnj2Wnn_Cij_sum_ij
        self.C0_mnj2Hnn_Cij_sum_ij = C0_mnj2Hnn_Cij_sum_ij
        self.C0_mnimnjWnn_Cijc_sum_ij = C0_mnimnjWnn_Cijc_sum_ij
        self.C0_mnimnjHnn_Cijc_sum_ij = C0_mnimnjHnn_Cijc_sum_ij
        self.C0_mni2mnj2Hnn_Cij_sum_ij = C0_mni2mnj2Hnn_Cij_sum_ij
        self.C0_mnimnj3Hnn_Cijc_sum_ij = C0_mnimnj3Hnn_Cijc_sum_ij
        self.C0_mni3mnjHnn_Cijc_sum_ij = C0_mni3mnjHnn_Cijc_sum_ij
        ########## One neutrino ###############################
        self.B11_nW_UU_sum_i_a = B11_nW_UU_sum_i_a
        self.B21_nW_UU_sum_i_b = B21_nW_UU_sum_i_b
        self.B10_nW_mni2_UU_sum_i_a = B10_nW_mni2_UU_sum_i_a
        self.B20_nW_mni2_UU_sum_i_b = B20_nW_mni2_UU_sum_i_b
        self.B11_nW_mni2_UU_sum_i_a = B11_nW_mni2_UU_sum_i_a
        self.B21_nW_mni2_UU_sum_i_b = B21_nW_mni2_UU_sum_i_b
        #########
        self.B11_nH_UU_sum_i_a = B11_nH_UU_sum_i_a
        self.B11_nH_mni2_UU_sum_i_a = B11_nH_mni2_UU_sum_i_a
        self.B10_nH_mni2_UU_sum_i_a = B10_nH_mni2_UU_sum_i_a
        self.B20_nH_mni2_UU_sum_i_b = B20_nH_mni2_UU_sum_i_b
        self.B21_nH_UU_sum_i_b = B21_nH_UU_sum_i_b
        self.B21_nH_mni2_UU_sum_i_b  = B21_nH_mni2_UU_sum_i_b

    
    def FFnH(self, tanb=tb0, cosab=cab0):

        mla = ml[self.a]
        mlb = ml[self.b]

        xi_lphi0 = xi_lphi(tanb, cosab)
        xi_lA0 = xi_lA(tanb)
        xi_nA0 = xi_nA(tanb)

        ALnH = (mla*mlb**2)/(mla**2 - mlb**2)*xi_lphi0*(
            xi_lA0**2*mla**2*self.B11_nH_UU_sum_i_a +
            xi_nA0**2*self.B11_nH_mni2_UU_sum_i_a +
            2*xi_lA0*xi_nA0*self.B10_nH_mni2_UU_sum_i_a
            )
        ARnH = (mlb)/(mla**2 - mlb**2)*xi_lphi0*(
            xi_lA0*xi_nA0*(mla**2 + mlb**2)*self.B10_nH_mni2_UU_sum_i_a +
            xi_lA0**2*mla**2*mlb**2*self.B11_nH_UU_sum_i_a +
            xi_nA0**2*mla**2*self.B11_nH_mni2_UU_sum_i_a
        )

        return ALnH, ARnH
    
    def FFHn(self, tanb=tb0, cosab=cab0):

        mla = ml[self.a]
        mlb = ml[self.b]

        xi_lphi0 = xi_lphi(tanb, cosab)
        xi_lA0 = xi_lA(tanb)
        xi_nA0 = xi_nA(tanb)

        ALHn = (mla)/(mla**2 - mlb**2)*xi_lphi0*(
            -xi_lA0*xi_nA0*(mla**2 + mlb**2)*self.B20_nH_mni2_UU_sum_i_b +
            xi_lA0**2*mla**2*mlb**2*self.B21_nH_UU_sum_i_b +
            xi_nA0**2*mlb**2*self.B21_nH_mni2_UU_sum_i_b
        )
        ARHn = (mla**2*mlb)/(mla**2 - mlb**2)*xi_lphi0*(
            xi_lA0**2*mlb**2*self.B21_nH_UU_sum_i_b +
            xi_nA0**2*self.B21_nH_mni2_UU_sum_i_b -
            2*xi_lA0*xi_nA0*self.B20_nH_mni2_UU_sum_i_b
        )

        return ALHn, ARHn

    def FFnG(self, tanb=tb0, cosab=cab0):

        mla = ml[self.a]
        mlb = ml[self.b]

        xi_lphi0 = xi_lphi(tanb, cosab)

        ALnG = (mla*mlb**2)/(mla**2 - mlb**2)*xi_lphi0*(
            - mla**2*self.B11_nW_UU_sum_i_a -
            self.B11_nW_mni2_UU_sum_i_a +
            2*self.B10_nW_mni2_UU_sum_i_a
            )
        ARnG = (mlb)/(mla**2 - mlb**2)*xi_lphi0*(
            (mla**2 + mlb**2)*self.B10_nW_mni2_UU_sum_i_a -
            mla**2*mlb**2*self.B11_nW_UU_sum_i_a -
            mla**2*self.B11_nW_mni2_UU_sum_i_a
        )

        return ALnG, ARnG

    def FFGn(self, tanb=tb0, cosab=cab0):

        mla = ml[self.a]
        mlb = ml[self.b]

        xi_lphi0 = xi_lphi(tanb, cosab)

        ALGn = -(mla)/(mla**2 - mlb**2)*xi_lphi0*(
            (mla**2 + mlb**2)*self.B20_nW_mni2_UU_sum_i_b +
            mla**2*mlb**2*self.B21_nW_UU_sum_i_b +
            mlb**2*self.B21_nW_mni2_UU_sum_i_b
        )
        ARGn = -(mla**2*mlb)/(mla**2 - mlb**2)*xi_lphi0*(
            mlb**2*self.B21_nW_UU_sum_i_b +
            self.B21_nW_mni2_UU_sum_i_b + 
            2*self.B20_nW_mni2_UU_sum_i_b
        )

        return ALGn, ARGn
    
    def FFnW(self, tanb=tb0, cosab=cab0):

        mla = ml[self.a]
        mlb = ml[self.b]

        xi_lphi0 = xi_lphi(tanb, cosab)

        ALnW = -(mW**2*mla*mlb**2)/(mla**2 - mlb**2)*xi_lphi0*self.B11_nW_UU_sum_i_a
        ARnW = -(mW**2*mla**2*mlb)/(mla**2 - mlb**2)*xi_lphi0*self.B11_nW_UU_sum_i_a

        return ALnW, ARnW
    
    def FFWn(self, tanb=tb0, cosab=cab0):

        mla = ml[self.a]
        mlb = ml[self.b]

        xi_lphi0 = xi_lphi(tanb, cosab)

        ALWn = -(mW**2*mla*mlb**2)/(mla**2 - mlb**2)*xi_lphi0*self.B21_nW_UU_sum_i_b
        ARWn = -(mW**2*mla**2*mlb)/(mla**2 - mlb**2)*xi_lphi0*self.B21_nW_UU_sum_i_b

        return ALWn, ARWn

    
    def FFGnn(self, tanb=tb0, cosab=cab0):
        
        mla = ml[self.a]
        mlb = ml[self.b]

        xi_nphi0 = xi_nphi(tanb, cosab)

        ALGninj = mla*xi_nphi0*F*(
            self.B012_mnj2_Cij_sum_ij + mW**2*self.C0_mnj2Wnn_Cij_sum_ij -
            (
                mla**2*self.C1_mnj2Wnn_Cij_sum_ij_a + mlb**2*self.C1_mni2Wnn_Cij_sum_ij_a + 2*self.C1_mni2mnj2Wnn_Cij_sum_ij_a
            ) + self.B012_mnimnj_Cijc_sum_ij + mW**2*self.C0_mnimnjWnn_Cijc_sum_ij - 
            (mla**2 + mlb**2)*self.C1_mnimnjWnn_Cijc_sum_ij_a + 
            self.mni2_mnj2_C1_mnimnjWnn_Cijc_sum_ij_a
        )

        ARGninj = mlb*xi_nphi0*F*(
            self.B012_mni2_Cij_sum_ij + mW**2*self.C0_mni2Wnn_Cij_sum_ij +
            (
                mla**2*self.C2_mnj2Wnn_Cij_sum_ij_b + mlb**2*self.C2_mni2Wnn_Cij_sum_ij_b - 2*self.C2_mni2mnj2Wnn_Cij_sum_ij_b
            ) + self.B012_mnimnj_Cijc_sum_ij + mW**2*self.C0_mnimnjWnn_Cijc_sum_ij +
            (mla**2 + mlb**2)*self.C2_mnimnjWnn_Cijc_sum_ij_b - 
            self.mni2_mnj2_C2_mnimnjWnn_Cijc_sum_ij_b
        )

        return ALGninj, ARGninj

    def FFWnn(self, tanb=tb0, cosab=cab0):

        mla = ml[self.a]
        mlb = ml[self.b]

        xi_nphi0 = xi_nphi(tanb, cosab)

        ALWninj = 2*mW**2*mla*xi_nphi0*F*(
            self.mni2_mnj2_C1_Wnn_Cijc_sum_ij_a - self.C0_mnj2Wnn_Cij_sum_ij -
            self.C0_mnimnjWnn_Cijc_sum_ij + 2*self.C1_mnimnjWnn_Cijc_sum_ij_a
        )

        ARWninj = - 2*mW**2*mlb*xi_nphi0*F*(
            self.mni2_mnj2_C2_Wnn_Cijc_sum_ij_b + self.C0_mni2Wnn_Cij_sum_ij +
            self.C0_mnimnjWnn_Cijc_sum_ij + 2*self.C2_mnimnjWnn_Cijc_sum_ij_b
        )

        return ALWninj, ARWninj

    def FFHnn(self, tanb=tb0, cosab=cab0):

        mla = ml[self.a]
        mlb = ml[self.b]

        xi_nphi0 = xi_nphi(tanb, cosab)
        xi_lA0 = xi_lA(tanb)
        xi_nA0 = xi_nA(tanb)

        ALHninj = mla*xi_nphi0*F*(
            xi_lA0*xi_nA0*self.B012_mnj2_Cij_sum_ij + mlb**2*xi_lA0*(xi_lA0 + xi_nA0)*(self.C2_mni2Hnn_Cij_sum_ij_b + self.C2_mnj2Hnn_Cij_sum_ij_b) -
            xi_nA0*xi_lA0*mla**2*self.C1_mnj2Hnn_Cij_sum_ij_a - xi_nA0*xi_lA0*mlb**2*self.C1_mni2Hnn_Cij_sum_ij_a -
            2*xi_nA0**2*self.C1_mni2mnj2Hnn_Cij_sum_ij_a + mlb**2*(xi_lA0**2 + xi_lA0*xi_nA0)*self.C0_mni2Hnn_Cij_sum_ij +
            xi_lA0*xi_nA0*mHpm**2*self.C0_mnj2Hnn_Cij_sum_ij + (xi_lA0*xi_nA0 + xi_nA0**2)*self.C0_mni2mnj2Hnn_Cij_sum_ij +
            #
            xi_lA0*xi_nA0*self.B012_mnimnj_Cijc_sum_ij + 2*xi_lA0*(xi_lA0 + xi_nA0)*mlb**2*self.C2_mnimnjHnn_Cijc_sum_ij_b -
            xi_lA0*xi_nA0*(mla**2 + mlb**2)*self.C1_mnimnjHnn_Cijc_sum_ij_a - xi_nA0**2*self.mni2_mnj2_C1_mnimnjHnn_Cijc_sum_ij_a +
            mlb**2*(xi_lA0**2 + xi_lA0*xi_nA0)*self.C0_mnimnjHnn_Cijc_sum_ij +
            xi_lA0*xi_nA0*mHpm**2*self.C0_mnimnjHnn_Cijc_sum_ij +
            (xi_lA0*xi_nA0 + xi_nA0**2)*self.C0_mnimnj3Hnn_Cijc_sum_ij
            )

        ARHninj = mlb*xi_nphi0*F*(
            xi_lA0*xi_nA0*self.B012_mni2_Cij_sum_ij - mla**2*xi_lA0*(xi_lA0 + xi_nA0)*(self.C1_mni2Hnn_Cij_sum_ij_a + self.C1_mnj2Hnn_Cij_sum_ij_a) -
            xi_nA0*xi_lA0*mla**2*self.C2_mnj2Hnn_Cij_sum_ij_b + xi_nA0*xi_lA0*mlb**2*self.C2_mni2Hnn_Cij_sum_ij_b +
            2*xi_nA0**2*self.C2_mni2mnj2Hnn_Cij_sum_ij_b + mla**2*(xi_lA0**2 + xi_lA0*xi_nA0)*self.C0_mnj2Hnn_Cij_sum_ij +
            xi_lA0*xi_nA0*mHpm**2*self.C0_mni2Hnn_Cij_sum_ij + (xi_lA0*xi_nA0 + xi_nA0**2)*self.C0_mni2mnj2Hnn_Cij_sum_ij +
            #
            xi_lA0*xi_nA0*self.B012_mnimnj_Cijc_sum_ij - 2*xi_lA0*(xi_lA0 + xi_nA0)*mla**2*self.C1_mnimnjHnn_Cijc_sum_ij_a -
            xi_lA0*xi_nA0*(mla**2 + mlb**2)*self.C2_mnimnjHnn_Cijc_sum_ij_b + xi_nA0**2*self.mni2_mnj2_C2_mnimnjHnn_Cijc_sum_ij_b +
            mla**2*(xi_lA0**2 + xi_lA0*xi_nA0)*self.C0_mnimnjHnn_Cijc_sum_ij +
            xi_lA0*xi_nA0*mHpm**2*self.C0_mnimnjHnn_Cijc_sum_ij +
            (xi_lA0*xi_nA0 + xi_nA0**2)*self.C0_mni3mnjHnn_Cijc_sum_ij
            )

        return ALHninj, ARHninj
    
    def formfactors_contributions(self, tanb=tb0, cosab=cab0):

        ALGnn, ARGnn = self.FFGnn(tanb, cosab)
        ALWnn, ARWnn = self.FFWnn(tanb, cosab)
        ALHnn, ARHnn = self.FFHnn(tanb, cosab)
        ALnG, ARnG = self.FFnG(tanb, cosab)
        ALGn, ARGn = self.FFGn(tanb, cosab)
        ALnW, ARnW = self.FFnW(tanb, cosab)
        ALWn, ARWn = self.FFWn(tanb, cosab)
        ALnH, ARnH = self.FFnH(tanb, cosab)
        ALHn, ARHn = self.FFHn(tanb, cosab)

        ALG = ALGnn + ALnG + ALGn 
        ARG = ARGnn + ARnG + ARGn 
        FFG = {'L': ALG, 'R': ARG}

        ALW = ALWnn + ALnW + ALWn
        ARW = ARWnn + ARnW + ARWn
        FFW = {'L': ALW, 'R': ARW}

        ALH = ALHnn + ALnH + ALHn
        ARH = ARHnn + ARnH + ARHn
        FFH = {'L': ALH, 'R': ARH}

        return FFG, FFW, FFH
    
    def total_formfactors(self, tanb=tb0, cosab=cab0):

        ALGnn, ARGnn = self.FFGnn(tanb, cosab)
        ALWnn, ARWnn = self.FFWnn(tanb, cosab)
        ALHnn, ARHnn = self.FFHnn(tanb, cosab)
        ALnG, ARnG = self.FFnG(tanb, cosab)
        ALGn, ARGn = self.FFGn(tanb, cosab)
        ALnW, ARnW = self.FFnW(tanb, cosab)
        ALWn, ARWn = self.FFWn(tanb, cosab)
        ALnH, ARnH = self.FFnH(tanb, cosab)
        ALHn, ARHn = self.FFHn(tanb, cosab)

        AL = ALGnn + ALWnn + ALHnn + ALnG + ALGn + ALnW + ALWn + ALnH + ALHn
        AR = ARGnn + ARWnn + ARHnn + ARnG + ARGn + ARnW + ARWn + ARnH + ARHn

        return AL, AR
    
    def Whll(self, tanb=tb0, cosab=cab0):

        mla = ml[self.a]
        mlb = ml[self.b]

        AL,AR = self.total_formfactors(tanb, cosab)
        return Γhlilj(AL, AR, mh, mla, mlb)

##########################
### a = 2, b = 3
##########################
Higgs_to_mu_tau = Higgs_to_ll(a=2, b=3,
    C1_mnimnjWnn_Cijc_sum_ij_a = C1_mnimnjWnn_Cijc_sum_23,
    C2_mnimnjWnn_Cijc_sum_ij_b = C2_mnimnjWnn_Cijc_sum_23,
    C2_mni2Hnn_Cij_sum_ij_b = C2_mni2Hnn_Cij_sum_23,
    C2_mnj2Hnn_Cij_sum_ij_b = C2_mnj2Hnn_Cij_sum_23,
    C1_mnj2Hnn_Cij_sum_ij_a = C1_mnj2Hnn_Cij_sum_23,
    C1_mni2Hnn_Cij_sum_ij_a = C1_mni2Hnn_Cij_sum_23,
    C2_mnimnjHnn_Cijc_sum_ij_b = C2_mnimnjHnn_Cijc_sum_23,
    C1_mnimnjHnn_Cijc_sum_ij_a = C1_mnimnjHnn_Cijc_sum_23,
    ################
    C1_mnj2Wnn_Cij_sum_ij_a = C1_mnj2Wnn_Cij_sum_23,
    C1_mni2Wnn_Cij_sum_ij_a = C1_mni2Wnn_Cij_sum_23,
    C1_mni2mnj2Wnn_Cij_sum_ij_a = C1_mni2mnj2Wnn_Cij_sum_23,
    mni2_mnj2_C1_mnimnjWnn_Cijc_sum_ij_a = mni2_mnj2_C1_mnimnjWnn_Cijc_sum_23,
    C2_mnj2Wnn_Cij_sum_ij_b = C2_mnj2Wnn_Cij_sum_23,
    C2_mni2Wnn_Cij_sum_ij_b = C2_mni2Wnn_Cij_sum_23,
    C2_mni2mnj2Wnn_Cij_sum_ij_b = C2_mni2mnj2Wnn_Cij_sum_23,
    mni2_mnj2_C2_mnimnjWnn_Cijc_sum_ij_b = mni2_mnj2_C2_mnimnjWnn_Cijc_sum_23,
    mni2_mnj2_C1_Wnn_Cijc_sum_ij_a = mni2_mnj2_C1_Wnn_Cijc_sum_23,
    mni2_mnj2_C2_Wnn_Cijc_sum_ij_b = mni2_mnj2_C2_Wnn_Cijc_sum_23,
    C1_mni2mnj2Hnn_Cij_sum_ij_a = C1_mni2mnj2Hnn_Cij_sum_23,
    mni2_mnj2_C1_mnimnjHnn_Cijc_sum_ij_a = mni2_mnj2_C1_mnimnjHnn_Cijc_sum_23,
    C2_mni2mnj2Hnn_Cij_sum_ij_b = C2_mni2mnj2Hnn_Cij_sum_23,
    mni2_mnj2_C2_mnimnjHnn_Cijc_sum_ij_b = mni2_mnj2_C2_mnimnjHnn_Cijc_sum_23,
    ############################ 
    B012_mni2_Cij_sum_ij = B012_mni2_Cij_sum_23,
    B012_mnj2_Cij_sum_ij = B012_mni2_Cij_sum_23,
    B012_mnimnj_Cijc_sum_ij = B012_mnimnj_Cijc_sum_23,
    C0_mni2Wnn_Cij_sum_ij = C0_mni2Wnn_Cij_sum_23,
    C0_mni2Hnn_Cij_sum_ij = C0_mni2Hnn_Cij_sum_23,
    C0_mnj2Wnn_Cij_sum_ij = C0_mnj2Wnn_Cij_sum_23,
    C0_mnj2Hnn_Cij_sum_ij = C0_mnj2Hnn_Cij_sum_23,
    C0_mnimnjWnn_Cijc_sum_ij = C0_mnimnjWnn_Cijc_sum_23,
    C0_mnimnjHnn_Cijc_sum_ij = C0_mnimnjHnn_Cijc_sum_23,
    C0_mni2mnj2Hnn_Cij_sum_ij = C0_mni2mnj2Hnn_Cij_sum_23,
    C0_mnimnj3Hnn_Cijc_sum_ij = C0_mnimnj3Hnn_Cijc_sum_23,
    C0_mni3mnjHnn_Cijc_sum_ij = C0_mni3mnjHnn_Cijc_sum_23,
    B11_nW_UU_sum_i_a = B11_nW_UU_sum_23,
    B21_nW_UU_sum_i_b = B21_nW_UU_sum_23,
    B10_nW_mni2_UU_sum_i_a = B10_nW_mni2_UU_sum_23,
    B20_nW_mni2_UU_sum_i_b = B20_nW_mni2_UU_sum_23,
    B11_nW_mni2_UU_sum_i_a = B11_nW_mni2_UU_sum_23,
    B21_nW_mni2_UU_sum_i_b = B21_nW_mni2_UU_sum_23,
    ########
    B11_nH_UU_sum_i_a = B11_nH_UU_sum_23,
    B11_nH_mni2_UU_sum_i_a = B11_nH_mni2_UU_sum_23,
    B10_nH_mni2_UU_sum_i_a = B10_nH_mni2_UU_sum_23,
    B20_nH_mni2_UU_sum_i_b = B20_nH_mni2_UU_sum_23,
    B21_nH_UU_sum_i_b = B21_nH_UU_sum_23,
    B21_nH_mni2_UU_sum_i_b = B21_nH_mni2_UU_sum_23
)

##########################
### a = 1, b = 3
##########################
Higgs_to_e_tau = Higgs_to_ll(a=1, b=3,
    C1_mnimnjWnn_Cijc_sum_ij_a = C1_mnimnjWnn_Cijc_sum_13,
    C2_mnimnjWnn_Cijc_sum_ij_b = C2_mnimnjWnn_Cijc_sum_13,
    C2_mni2Hnn_Cij_sum_ij_b = C2_mni2Hnn_Cij_sum_13,
    C2_mnj2Hnn_Cij_sum_ij_b = C2_mnj2Hnn_Cij_sum_13,
    C1_mnj2Hnn_Cij_sum_ij_a = C1_mnj2Hnn_Cij_sum_13,
    C1_mni2Hnn_Cij_sum_ij_a = C1_mni2Hnn_Cij_sum_13,
    C2_mnimnjHnn_Cijc_sum_ij_b = C2_mnimnjHnn_Cijc_sum_13,
    C1_mnimnjHnn_Cijc_sum_ij_a = C1_mnimnjHnn_Cijc_sum_13,
    ################
    C1_mnj2Wnn_Cij_sum_ij_a = C1_mnj2Wnn_Cij_sum_13,
    C1_mni2Wnn_Cij_sum_ij_a = C1_mni2Wnn_Cij_sum_13,
    C1_mni2mnj2Wnn_Cij_sum_ij_a = C1_mni2mnj2Wnn_Cij_sum_13,
    mni2_mnj2_C1_mnimnjWnn_Cijc_sum_ij_a = mni2_mnj2_C1_mnimnjWnn_Cijc_sum_13,
    C2_mnj2Wnn_Cij_sum_ij_b = C2_mnj2Wnn_Cij_sum_13,
    C2_mni2Wnn_Cij_sum_ij_b = C2_mni2Wnn_Cij_sum_13,
    C2_mni2mnj2Wnn_Cij_sum_ij_b = C2_mni2mnj2Wnn_Cij_sum_13,
    mni2_mnj2_C2_mnimnjWnn_Cijc_sum_ij_b = mni2_mnj2_C2_mnimnjWnn_Cijc_sum_13,
    mni2_mnj2_C1_Wnn_Cijc_sum_ij_a = mni2_mnj2_C1_Wnn_Cijc_sum_13,
    mni2_mnj2_C2_Wnn_Cijc_sum_ij_b = mni2_mnj2_C2_Wnn_Cijc_sum_13,
    C1_mni2mnj2Hnn_Cij_sum_ij_a = C1_mni2mnj2Hnn_Cij_sum_13,
    mni2_mnj2_C1_mnimnjHnn_Cijc_sum_ij_a = mni2_mnj2_C1_mnimnjHnn_Cijc_sum_13,
    C2_mni2mnj2Hnn_Cij_sum_ij_b = C2_mni2mnj2Hnn_Cij_sum_13,
    mni2_mnj2_C2_mnimnjHnn_Cijc_sum_ij_b = mni2_mnj2_C2_mnimnjHnn_Cijc_sum_13,
    ############################ 
    B012_mni2_Cij_sum_ij = B012_mni2_Cij_sum_13,
    B012_mnj2_Cij_sum_ij = B012_mni2_Cij_sum_13,
    B012_mnimnj_Cijc_sum_ij = B012_mnimnj_Cijc_sum_13,
    C0_mni2Wnn_Cij_sum_ij = C0_mni2Wnn_Cij_sum_13,
    C0_mni2Hnn_Cij_sum_ij = C0_mni2Hnn_Cij_sum_13,
    C0_mnj2Wnn_Cij_sum_ij = C0_mnj2Wnn_Cij_sum_13,
    C0_mnj2Hnn_Cij_sum_ij = C0_mnj2Hnn_Cij_sum_13,
    C0_mnimnjWnn_Cijc_sum_ij = C0_mnimnjWnn_Cijc_sum_13,
    C0_mnimnjHnn_Cijc_sum_ij = C0_mnimnjHnn_Cijc_sum_13,
    C0_mni2mnj2Hnn_Cij_sum_ij = C0_mni2mnj2Hnn_Cij_sum_13,
    C0_mnimnj3Hnn_Cijc_sum_ij = C0_mnimnj3Hnn_Cijc_sum_13,
    C0_mni3mnjHnn_Cijc_sum_ij = C0_mni3mnjHnn_Cijc_sum_13,
    B11_nW_UU_sum_i_a = B11_nW_UU_sum_13,
    B21_nW_UU_sum_i_b = B21_nW_UU_sum_13,
    B10_nW_mni2_UU_sum_i_a = B10_nW_mni2_UU_sum_13,
    B20_nW_mni2_UU_sum_i_b = B20_nW_mni2_UU_sum_13,
    B11_nW_mni2_UU_sum_i_a = B11_nW_mni2_UU_sum_13,
    B21_nW_mni2_UU_sum_i_b = B21_nW_mni2_UU_sum_13,
    ########
    B11_nH_UU_sum_i_a = B11_nH_UU_sum_13,
    B11_nH_mni2_UU_sum_i_a = B11_nH_mni2_UU_sum_13,
    B10_nH_mni2_UU_sum_i_a = B10_nH_mni2_UU_sum_13,
    B20_nH_mni2_UU_sum_i_b = B20_nH_mni2_UU_sum_13,
    B21_nH_UU_sum_i_b = B21_nH_UU_sum_13,
    B21_nH_mni2_UU_sum_i_b = B21_nH_mni2_UU_sum_13
)

##########################
### a = 1, b = 2
##########################
Higgs_to_e_mu = Higgs_to_ll(a=1, b=2,
    C1_mnimnjWnn_Cijc_sum_ij_a = C1_mnimnjWnn_Cijc_sum_12,
    C2_mnimnjWnn_Cijc_sum_ij_b = C2_mnimnjWnn_Cijc_sum_12,
    C2_mni2Hnn_Cij_sum_ij_b = C2_mni2Hnn_Cij_sum_12,
    C2_mnj2Hnn_Cij_sum_ij_b = C2_mnj2Hnn_Cij_sum_12,
    C1_mnj2Hnn_Cij_sum_ij_a = C1_mnj2Hnn_Cij_sum_12,
    C1_mni2Hnn_Cij_sum_ij_a = C1_mni2Hnn_Cij_sum_12,
    C2_mnimnjHnn_Cijc_sum_ij_b = C2_mnimnjHnn_Cijc_sum_12,
    C1_mnimnjHnn_Cijc_sum_ij_a = C1_mnimnjHnn_Cijc_sum_12,
    ################
    C1_mnj2Wnn_Cij_sum_ij_a = C1_mnj2Wnn_Cij_sum_12,
    C1_mni2Wnn_Cij_sum_ij_a = C1_mni2Wnn_Cij_sum_12,
    C1_mni2mnj2Wnn_Cij_sum_ij_a = C1_mni2mnj2Wnn_Cij_sum_12,
    mni2_mnj2_C1_mnimnjWnn_Cijc_sum_ij_a = mni2_mnj2_C1_mnimnjWnn_Cijc_sum_12,
    C2_mnj2Wnn_Cij_sum_ij_b = C2_mnj2Wnn_Cij_sum_12,
    C2_mni2Wnn_Cij_sum_ij_b = C2_mni2Wnn_Cij_sum_12,
    C2_mni2mnj2Wnn_Cij_sum_ij_b = C2_mni2mnj2Wnn_Cij_sum_12,
    mni2_mnj2_C2_mnimnjWnn_Cijc_sum_ij_b = mni2_mnj2_C2_mnimnjWnn_Cijc_sum_12,
    mni2_mnj2_C1_Wnn_Cijc_sum_ij_a = mni2_mnj2_C1_Wnn_Cijc_sum_12,
    mni2_mnj2_C2_Wnn_Cijc_sum_ij_b = mni2_mnj2_C2_Wnn_Cijc_sum_12,
    C1_mni2mnj2Hnn_Cij_sum_ij_a = C1_mni2mnj2Hnn_Cij_sum_12,
    mni2_mnj2_C1_mnimnjHnn_Cijc_sum_ij_a = mni2_mnj2_C1_mnimnjHnn_Cijc_sum_12,
    C2_mni2mnj2Hnn_Cij_sum_ij_b = C2_mni2mnj2Hnn_Cij_sum_12,
    mni2_mnj2_C2_mnimnjHnn_Cijc_sum_ij_b = mni2_mnj2_C2_mnimnjHnn_Cijc_sum_12,
    ############################ 
    B012_mni2_Cij_sum_ij = B012_mni2_Cij_sum_12,
    B012_mnj2_Cij_sum_ij = B012_mni2_Cij_sum_12,
    B012_mnimnj_Cijc_sum_ij = B012_mnimnj_Cijc_sum_12,
    C0_mni2Wnn_Cij_sum_ij = C0_mni2Wnn_Cij_sum_12,
    C0_mni2Hnn_Cij_sum_ij = C0_mni2Hnn_Cij_sum_12,
    C0_mnj2Wnn_Cij_sum_ij = C0_mnj2Wnn_Cij_sum_12,
    C0_mnj2Hnn_Cij_sum_ij = C0_mnj2Hnn_Cij_sum_12,
    C0_mnimnjWnn_Cijc_sum_ij = C0_mnimnjWnn_Cijc_sum_12,
    C0_mnimnjHnn_Cijc_sum_ij = C0_mnimnjHnn_Cijc_sum_12,
    C0_mni2mnj2Hnn_Cij_sum_ij = C0_mni2mnj2Hnn_Cij_sum_12,
    C0_mnimnj3Hnn_Cijc_sum_ij = C0_mnimnj3Hnn_Cijc_sum_12,
    C0_mni3mnjHnn_Cijc_sum_ij = C0_mni3mnjHnn_Cijc_sum_12,
    B11_nW_UU_sum_i_a = B11_nW_UU_sum_12,
    B21_nW_UU_sum_i_b = B21_nW_UU_sum_12,
    B10_nW_mni2_UU_sum_i_a = B10_nW_mni2_UU_sum_12,
    B20_nW_mni2_UU_sum_i_b = B20_nW_mni2_UU_sum_12,
    B11_nW_mni2_UU_sum_i_a = B11_nW_mni2_UU_sum_12,
    B21_nW_mni2_UU_sum_i_b = B21_nW_mni2_UU_sum_12,
    ########
    B11_nH_UU_sum_i_a = B11_nH_UU_sum_12,
    B11_nH_mni2_UU_sum_i_a = B11_nH_mni2_UU_sum_12,
    B10_nH_mni2_UU_sum_i_a = B10_nH_mni2_UU_sum_12,
    B20_nH_mni2_UU_sum_i_b = B20_nH_mni2_UU_sum_12,
    B21_nH_UU_sum_i_b = B21_nH_UU_sum_12,
    B21_nH_mni2_UU_sum_i_b = B21_nH_mni2_UU_sum_12

)

class LFV_HD():
    def __init__(self, h23, h13, h12):
        self.h23 = h23
        self.h13 = h13
        self.h12 = h12

    def total_width(self, tanb=tb0, cosab=cab0):
        Wh23 = self.h23.Whll(tanb, cosab)
        Wh13 = self.h13.Whll(tanb, cosab)
        Wh12 = self.h12.Whll(tanb, cosab)
    
        return Wh23 + Wh13 + Wh12 + 0.0032
    
    def BRh23(self, tanb=tb0, cosab=cab0):
        return self.h23.Whll(tanb, cosab)/self.total_width(tanb, cosab)
    
    def BRh13(self, tanb=tb0, cosab=cab0):
        return self.h13.Whll(tanb, cosab)/self.total_width(tanb, cosab)
    
    def BRh12(self, tanb=tb0, cosab=cab0):
        return self.h12.Whll(tanb, cosab)/self.total_width(tanb, cosab)

lfvhd = LFV_HD(Higgs_to_mu_tau, Higgs_to_e_tau, Higgs_to_e_mu)

if __name__ == '__main__':
    from time import perf_counter
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    from multiprocessing import Pool, cpu_count

    def pretty(x):
        nprint(chop(x))
    
    start = perf_counter()

    print('Branching ratios: ')
    nprint(chop(lfvhd.BRh23()))
    nprint(chop(lfvhd.BRh13()))
    nprint(chop(lfvhd.BRh12()))
    
    tbi = -1
    tbf = 2
    n = 100
    expmp = linspace(tbi, tbf,n)
    tbmp = np.array([mpf('10.0')**k for k in expmp])

    #Whll = []
    #for tba1 in tbmp:
    #    W =  Higgs_to_mu_tau.Whll(tba1, mpf('1e-2'))
    #    Whll.append(W)
    #Whll_np = np.array(Whll)

    # cab0 = mpf('1e-2')
    BRh23 = np.array([lfvhd.BRh23(tb1, cosab=cab0) for tb1 in tbmp])
    BRh13 = np.array([lfvhd.BRh13(tb1, cosab=cab0) for tb1 in tbmp])
    BRh12 = np.array([lfvhd.BRh12(tb1, cosab=cab0) for tb1 in tbmp])

    xi_nphi_mp = np.array([(xi_nphi(tb1, cab0)*xi_lA(tb1)*xi_nA(tb1))**2 for tb1 in tbmp])

    plt.figure(figsize=(12,9))
    #plt.loglog(tbmp, Whll_np, '-', label=r'$|\Gamma((h \to \mu \tau)|^2$')

    plt.loglog(tbmp, BRh23, '-', label=r'$\mathcal{BR}(h \to \mu \tau)$')
    plt.loglog(tbmp, BRh13, '-', label=r'$\mathcal{BR}(h \to e \tau)$')
    plt.loglog(tbmp, BRh12, '-', label=r'$\mathcal{BR}(h \to e \mu)$')
    #plt.loglog(tbmp, xi_nphi_mp, '--', label=r'$(\xi_h^n \xi_A^\ell \xi_A^n)^2$')
    #plt.axvspan(xmin=1e-1,xmax=1,ymin=1e-12,ymax=1e-3,alpha=0.5)
    plt.axvspan(1e-1,1,alpha=0.5)
    plt.text(s="Perturvatividad",x=2e-1,y=5e-10,fontsize=35,rotation=90,alpha=0.6)

    s = 25
    plt.xlabel(r'$\tan{\beta}$',fontsize=s)
    plt.ylabel(r'$\mathcal{BR}(h \to \ell_a \ell_b)$',fontsize=s)
    plt.tick_params(labelsize=s)
    plt.xlim(1e-1,1e2)
    plt.legend(fontsize=s)
    #plt.show()

    n = 1000
    X1 = np.random.uniform(tbi, tbf, n)
    X = np.array([mpf('10.0')**k for k in X1])
    Y1 = np.random.uniform(-4, 0, n)
    Y = np.array([mpf('10.0')**k for k in Y1])

    Z = np.array(
        [lfvhd.BRh23(x, y) for x, y in zip(X,Y)]
    )

    plt.figure()
    plt.scatter(
        X, Y, c=Z,
        norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
        edgecolors=None
    )
    
    plt.colorbar()
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r'$\tan{\beta}$')
    plt.ylabel(r'$\cos{\beta - \alpha}$')
    plt.show()

    end = perf_counter()
    
    print('EL tiempo de ejecución es: \n')
    print((end - start)/60**2)