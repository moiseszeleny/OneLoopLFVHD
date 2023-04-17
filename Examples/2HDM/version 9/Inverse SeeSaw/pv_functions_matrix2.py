### Neutrino benchmark
from OneLoopLFVHD.neutrinos import NuOscObservables

Nudata = NuOscObservables

from mpmath import *

mp.dps = 32; mp.pretty = True


m1 = mpf('1e-12')  #GeV 

#current values to Square mass differences
d21 = mpf(str(Nudata.squareDm21.central))*mpf('1e-18')# factor to convert eV^2 to GeV^2
d31 = mpf(str(Nudata.squareDm31.central))*mpf('1e-18')

#d21 = 7.5e-5*1e-18
#d31 = 2.457e-3*1e-18
m2 = sqrt(m1**2 + d21)
m3 = sqrt(m1**2 + d31)


### Diagonalización
from Unu_seesaw_ISS import diagonalizationMnu_ISS as diagonalizationMnu

### Numeric translation 
from OneLoopLFVHD.data import ml

###### 
# Caso degenerado
######
diagonalizationMnu1 = lambda m1, m6, mux: diagonalizationMnu(
    m1, m2, m3, m6, m6, m6, mux, mux, mux)

MR = mpf('1e4')
mux = mpf('1e-6')
mHpm = mpf('300')
mW = mpf('80.379')
mh = mpf('125.1')

mn,UnuL,UnuR = diagonalizationMnu1(m1, MR, mux)

a = 2
b = 3
Unu = UnuL
Unu_dagger = UnuR

Cij = lambda i,j: mp.fsum([UnuL[c,i]*UnuR[j,c] for c in range(3)])

import OneLoopLFVHD.LFVHDFeynG_mpmath2 as lfvhd_mp

##################################
######## Wninj
##################################

# B012 matrix ninj 
B012_nn = matrix(
    [[
        lfvhd_mp.B12_0(mh, mn[i-1],mn[j-1]) for i in range(9)
        ] for j in range(9)]
    )

def B012_mni2_Cij_sum(a,b):
    return fsum(
    [
        B012_nn[i-1, j-1]*mn[i-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

B012_mni2_Cij_sum_23 = B012_mni2_Cij_sum(2,3)
B012_mni2_Cij_sum_13 = B012_mni2_Cij_sum(1,3)
B012_mni2_Cij_sum_12 = B012_mni2_Cij_sum(1,2)


def B012_mnj2_Cij_sum(a,b):
    return fsum(
    [
        B012_nn[i-1, j-1]*mn[j-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

B012_mnj2_Cij_sum_23 = B012_mnj2_Cij_sum(2, 3)
B012_mnj2_Cij_sum_13 = B012_mnj2_Cij_sum(1, 3)
B012_mnj2_Cij_sum_12 = B012_mnj2_Cij_sum(1, 2)

def B012_mnimnj_Cijc_sum(a,b):
    return fsum(
    [
        B012_nn[i-1, j-1]*mn[i-1]*mn[j-1]*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

B012_mnimnj_Cijc_sum_23 = B012_mnimnj_Cijc_sum(2,3)
B012_mnimnj_Cijc_sum_13 = B012_mnimnj_Cijc_sum(1,3)
B012_mnimnj_Cijc_sum_12 = B012_mnimnj_Cijc_sum(1,2)

# C0 matrix Wninj
C0_Wnn = matrix([
    [
        lfvhd_mp.C0(mh,mW,mn[i],mn[j]) for i in range(9)
        ] for j in range(9)
        ]
    )

# C0 matrix Hninj
C0_Hnn = matrix([
    [
        lfvhd_mp.C0(mh, mHpm, mn[i], mn[j]) for i in range(9)
        ] for j in range(9)
        ]
    )

# C0(Wninj) mni^2
def C0_mni2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C0_Wnn[i-1,j-1]*mn[i-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C0_mni2Wnn_Cij_sum_23 = C0_mni2Wnn_Cij_sum(2, 3)
C0_mni2Wnn_Cij_sum_13 = C0_mni2Wnn_Cij_sum(1, 3)
C0_mni2Wnn_Cij_sum_12 = C0_mni2Wnn_Cij_sum(1, 2)

# C0(Hninj) mni^2
def C0_mni2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C0_Hnn[i-1,j-1]*mn[i-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C0_mni2Hnn_Cij_sum_23 = C0_mni2Hnn_Cij_sum(2, 3)
C0_mni2Hnn_Cij_sum_13 = C0_mni2Hnn_Cij_sum(1, 3)
C0_mni2Hnn_Cij_sum_12 = C0_mni2Hnn_Cij_sum(1, 2)

# C0(Wninj) mnj^2
def C0_mnj2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C0_Wnn[i-1,j-1]*mn[j-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C0_mnj2Wnn_Cij_sum_23 = C0_mnj2Wnn_Cij_sum(2, 3)
C0_mnj2Wnn_Cij_sum_13 = C0_mnj2Wnn_Cij_sum(1, 3)
C0_mnj2Wnn_Cij_sum_12 = C0_mnj2Wnn_Cij_sum(1, 2)

# C0(Hninj) mnj^2
def C0_mnj2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C0_Hnn[i-1,j-1]*mn[j-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C0_mnj2Hnn_Cij_sum_23 = C0_mnj2Hnn_Cij_sum(2, 3)
C0_mnj2Hnn_Cij_sum_13 = C0_mnj2Hnn_Cij_sum(1, 3)
C0_mnj2Hnn_Cij_sum_12 = C0_mnj2Hnn_Cij_sum(1, 2)

# C0(Wninj) mni mnj Cijc
def C0_mnimnjWnn_Cijc_sum(a, b):
    return fsum(
    [
        C0_Wnn[i-1,j-1]*mn[i-1]*mn[j-1]*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, 10) for j in range(1, 10)
        ]
    )

C0_mnimnjWnn_Cijc_sum_23 = C0_mnimnjWnn_Cijc_sum(2, 3)
C0_mnimnjWnn_Cijc_sum_13 = C0_mnimnjWnn_Cijc_sum(1, 3)
C0_mnimnjWnn_Cijc_sum_12 = C0_mnimnjWnn_Cijc_sum(1, 2)

# C0(Hninj) mni mnj Cijc
def C0_mnimnjHnn_Cijc_sum(a, b):
    return fsum(
    [
        C0_Hnn[i-1,j-1]*mn[i-1]*mn[j-1]*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, 10) for j in range(1, 10)
        ]
    )

C0_mnimnjHnn_Cijc_sum_23 = C0_mnimnjHnn_Cijc_sum(2, 3)
C0_mnimnjHnn_Cijc_sum_13 = C0_mnimnjHnn_Cijc_sum(1, 3)
C0_mnimnjHnn_Cijc_sum_12 = C0_mnimnjHnn_Cijc_sum(1, 2)

# C0(Hninj) mni^2 mnj^2
def C0_mni2mnj2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C0_Hnn[i-1,j-1]*mn[i-1]**2*mn[j-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, 10) for j in range(1, 10)
        ]
    )

C0_mni2mnj2Hnn_Cij_sum_23 = C0_mni2mnj2Hnn_Cij_sum(2, 3)
C0_mni2mnj2Hnn_Cij_sum_13 = C0_mni2mnj2Hnn_Cij_sum(1, 3)
C0_mni2mnj2Hnn_Cij_sum_12 = C0_mni2mnj2Hnn_Cij_sum(1, 2)


# C0(Hninj) mni mnj^3 Cijc
def C0_mnimnj3Hnn_Cijc_sum(a, b):
    return fsum(
    [
        C0_Hnn[i-1,j-1]*mn[i-1]*mn[j-1]**3*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, 10) for j in range(1, 10)
        ]
    )

C0_mnimnj3Hnn_Cijc_sum_23 = C0_mnimnj3Hnn_Cijc_sum(2, 3)
C0_mnimnj3Hnn_Cijc_sum_13 = C0_mnimnj3Hnn_Cijc_sum(1, 3)
C0_mnimnj3Hnn_Cijc_sum_12 = C0_mnimnj3Hnn_Cijc_sum(1, 2)

# C0(Hninj) mni^3 mnj Cijc 
def C0_mni3mnjHnn_Cijc_sum(a, b):
    return fsum(
    [
        C0_Hnn[i-1,j-1]*mn[i-1]**3*mn[j-1]*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1]
        for i in range(1, 10) for j in range(1, 10)
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
            lfvhd_mp.C1(mh, ml[a], mW, mn[i], mn[j]) for i in range(9)
            ] for j in range(9)
            ]
        )

C1_Wnn_2 = C1_Wnn(2)
C1_Wnn_1 = C1_Wnn(1)

def C1_Wnn_caso(a):
    if a == 1:
        C1_Wnn_a = C1_Wnn_1
    else:
        C1_Wnn_a = C1_Wnn_2
    
    return C1_Wnn_a

# C1 matrix Hninj
def C1_Hnn(a):
    return matrix([
        [
            lfvhd_mp.C1(mh, ml[a], mHpm, mn[i], mn[j]) for i in range(9)
            ] for j in range(9)
        ]
        )

C1_Hnn_2 = C1_Hnn(2)
C1_Hnn_1 = C1_Hnn(1) 

def C1_Hnn_caso(a):
    if a == 1:
        C1_Hnn_a = C1_Wnn_1
    else:
        C1_Hnn_a = C1_Wnn_2
    
    return C1_Hnn_a

# C1(Wninj)mnj^2
def C1_mnj2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C1_Wnn_caso(a)[i-1,j-1]*mn[j-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )
C1_mnj2Wnn_Cij_sum_23 = C1_mnj2Wnn_Cij_sum(2, 3)
C1_mnj2Wnn_Cij_sum_13 = C1_mnj2Wnn_Cij_sum(1, 3)
C1_mnj2Wnn_Cij_sum_12 = C1_mnj2Wnn_Cij_sum(1, 2)

# C1(Hninj)mnj^2
def C1_mnj2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C1_Hnn_caso(a)[i-1,j-1]*mn[j-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C1_mnj2Hnn_Cij_sum_23 = C1_mnj2Hnn_Cij_sum(2, 3)
C1_mnj2Hnn_Cij_sum_13 = C1_mnj2Hnn_Cij_sum(1, 3)
C1_mnj2Hnn_Cij_sum_12 = C1_mnj2Hnn_Cij_sum(1, 2)

# C1(Wninj)mni^2
def C1_mni2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C1_Wnn_caso(a)[i-1,j-1]*mn[i-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C1_mni2Wnn_Cij_sum_23 = C1_mni2Wnn_Cij_sum(2, 3)
C1_mni2Wnn_Cij_sum_13 = C1_mni2Wnn_Cij_sum(1, 3)
C1_mni2Wnn_Cij_sum_12 = C1_mni2Wnn_Cij_sum(1, 2)

# C1(Hninj)mni^2
def C1_mni2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C1_Hnn_caso(a)[i-1,j-1]*mn[i-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C1_mni2Hnn_Cij_sum_23 = C1_mni2Hnn_Cij_sum(2, 3)
C1_mni2Hnn_Cij_sum_13 = C1_mni2Hnn_Cij_sum(1, 3)
C1_mni2Hnn_Cij_sum_12 = C1_mni2Hnn_Cij_sum(1, 2)

# C1(Wninj)mni^2 mnj^2
def C1_mni2mnj2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C1_Wnn_caso(a)[i-1,j-1]*mn[i-1]**2*mn[j-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C1_mni2mnj2Wnn_Cij_sum_23 = C1_mni2mnj2Wnn_Cij_sum(2,3)
C1_mni2mnj2Wnn_Cij_sum_13 = C1_mni2mnj2Wnn_Cij_sum(1,3)
C1_mni2mnj2Wnn_Cij_sum_12 = C1_mni2mnj2Wnn_Cij_sum(1,2)

# C1(Hninj)mni^2 mnj^2
def C1_mni2mnj2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C1_Hnn_caso(a)[i-1,j-1]*mn[i-1]**2*mn[j-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C1_mni2mnj2Hnn_Cij_sum_23 = C1_mni2mnj2Hnn_Cij_sum(2, 3)
C1_mni2mnj2Hnn_Cij_sum_13 = C1_mni2mnj2Hnn_Cij_sum(1, 3)
C1_mni2mnj2Hnn_Cij_sum_12 = C1_mni2mnj2Hnn_Cij_sum(1, 2)

# C1(Wninj)mni mnj Cijc
def C1_mnimnjWnn_Cijc_sum(a, b):
    return fsum(
    [
        C1_Wnn_caso(a)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C1_mnimnjWnn_Cijc_sum_23 = C1_mnimnjWnn_Cijc_sum(2, 3)
C1_mnimnjWnn_Cijc_sum_13 = C1_mnimnjWnn_Cijc_sum(1, 3)
C1_mnimnjWnn_Cijc_sum_12 = C1_mnimnjWnn_Cijc_sum(1, 2)

# C1(Hninj)mni mnj Cijc
def C1_mnimnjHnn_Cijc_sum(a, b):
    return fsum(
    [
        C1_Hnn_caso(a)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C1_mnimnjHnn_Cijc_sum_23 = C1_mnimnjHnn_Cijc_sum(2, 3)
C1_mnimnjHnn_Cijc_sum_13 = C1_mnimnjHnn_Cijc_sum(1, 3)
C1_mnimnjHnn_Cijc_sum_12 = C1_mnimnjHnn_Cijc_sum(1, 2)

# (mni^2 + mnj^2) C1(Wninj)
def mni2_mnj2_C1_Wnn_Cijc_sum(a, b):
    return fsum(
    [
        (mn[i-1]**2 + mn[j-1]**2)*C1_Wnn_caso(a)[i-1,j-1]*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

mni2_mnj2_C1_Wnn_Cijc_sum_23 = mni2_mnj2_C1_Wnn_Cijc_sum(2, 3)
mni2_mnj2_C1_Wnn_Cijc_sum_13 = mni2_mnj2_C1_Wnn_Cijc_sum(1, 3)
mni2_mnj2_C1_Wnn_Cijc_sum_12 = mni2_mnj2_C1_Wnn_Cijc_sum(1, 2)

# (mni^2 + mnj^2) C1(Wninj) mni mnj Cijc
def mni2_mnj2_C1_mnimnjWnn_Cijc_sum(a, b):
    return fsum(
    [
        (mn[i-1]**2 + mn[j-1]**2)*C1_Wnn_caso(a)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

mni2_mnj2_C1_mnimnjWnn_Cijc_sum_23 = mni2_mnj2_C1_mnimnjWnn_Cijc_sum(2, 3)
mni2_mnj2_C1_mnimnjWnn_Cijc_sum_13 = mni2_mnj2_C1_mnimnjWnn_Cijc_sum(1, 3)
mni2_mnj2_C1_mnimnjWnn_Cijc_sum_12 = mni2_mnj2_C1_mnimnjWnn_Cijc_sum(1, 2)

# (mni^2 + mnj^2) C1(Hninj) mni mnj Cijc
def mni2_mnj2_C1_mnimnjHnn_Cijc_sum(a, b):
    return fsum(
    [
        (mn[i-1]**2 + mn[j-1]**2)*C1_Hnn_caso(a)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
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
        lfvhd_mp.C2(mh, ml[b], mW, mn[i], mn[j]) for i in range(9)
        ] for j in range(9)
        ]
)

C2_Wnn_2 = C2_Wnn(2)
C2_Wnn_1 = C2_Wnn(1)

def C2_Wnn_caso(b):
    if b == 1:
        C2_Wnn_b = C2_Wnn_1
    else:
        C2_Wnn_b = C2_Wnn_2
    
    return C2_Wnn_b

# C2 matrix Hninj
def C2_Hnn(b):
    return matrix([
    [
        lfvhd_mp.C2(mh, ml[b], mHpm, mn[i], mn[j]) for i in range(9)
        ] for j in range(9)
        ]
)

C2_Hnn_2 = C2_Hnn(2)
C2_Hnn_1 = C2_Hnn(1)

def C2_Hnn_caso(b):
    if b == 1:
        C2_Hnn_b = C2_Hnn_1
    else:
        C2_Hnn_b = C2_Hnn_2
    
    return C2_Hnn_b

# C2(Wninj)*mnj^2
def C2_mnj2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C2_Wnn_caso(b)[i-1,j-1]*mn[j-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C2_mnj2Wnn_Cij_sum_23 = C2_mnj2Wnn_Cij_sum(2, 3)
C2_mnj2Wnn_Cij_sum_13 = C2_mnj2Wnn_Cij_sum(1, 3)
C2_mnj2Wnn_Cij_sum_12 = C2_mnj2Wnn_Cij_sum(1, 2)

# C2(Hninj)*mnj^2
def C2_mnj2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C2_Hnn_caso(b)[i-1,j-1]*mn[j-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C2_mnj2Hnn_Cij_sum_23 = C2_mnj2Hnn_Cij_sum(2, 3)
C2_mnj2Hnn_Cij_sum_13 = C2_mnj2Hnn_Cij_sum(1, 3)
C2_mnj2Hnn_Cij_sum_12 = C2_mnj2Hnn_Cij_sum(1, 2)

# C2(Wninj)*mni^2
def C2_mni2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C2_Wnn_caso(b)[i-1,j-1]*mn[i-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C2_mni2Wnn_Cij_sum_23 = C2_mni2Wnn_Cij_sum(2, 3)
C2_mni2Wnn_Cij_sum_13 = C2_mni2Wnn_Cij_sum(1, 3)
C2_mni2Wnn_Cij_sum_12 = C2_mni2Wnn_Cij_sum(1, 2)

# C2(Hninj)*mni^2
def C2_mni2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C2_Hnn_caso(b)[i-1,j-1]*mn[i-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C2_mni2Hnn_Cij_sum_23 = C2_mni2Hnn_Cij_sum(2, 3)
C2_mni2Hnn_Cij_sum_13 = C2_mni2Hnn_Cij_sum(1, 3)
C2_mni2Hnn_Cij_sum_12 = C2_mni2Hnn_Cij_sum(1, 2)

# C2(Wninj)*mni^2*mnj^2
def C2_mni2mnj2Wnn_Cij_sum(a, b):
    return fsum(
    [
        C2_Wnn_caso(b)[i-1,j-1]*mn[i-1]**2*mn[j-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C2_mni2mnj2Wnn_Cij_sum_23 = C2_mni2mnj2Wnn_Cij_sum(2, 3)
C2_mni2mnj2Wnn_Cij_sum_13 = C2_mni2mnj2Wnn_Cij_sum(1, 3)
C2_mni2mnj2Wnn_Cij_sum_12 = C2_mni2mnj2Wnn_Cij_sum(1, 2)

# C2(Hninj)*mni^2*mnj^2
def C2_mni2mnj2Hnn_Cij_sum(a, b):
    return fsum(
    [
        C2_Hnn_caso(b)[i-1,j-1]*mn[i-1]**2*mn[j-1]**2*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C2_mni2mnj2Hnn_Cij_sum_23 = C2_mni2mnj2Hnn_Cij_sum(2, 3)
C2_mni2mnj2Hnn_Cij_sum_13 = C2_mni2mnj2Hnn_Cij_sum(1, 3)
C2_mni2mnj2Hnn_Cij_sum_12 = C2_mni2mnj2Hnn_Cij_sum(1, 2)

# C2(Wninj)*mni*mnj Cijc
def C2_mnimnjWnn_Cijc_sum(a, b):
    return fsum(
    [
        C2_Wnn_caso(b)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C2_mnimnjWnn_Cijc_sum_23 = C2_mnimnjWnn_Cijc_sum(2, 3)
C2_mnimnjWnn_Cijc_sum_13 = C2_mnimnjWnn_Cijc_sum(1, 3)
C2_mnimnjWnn_Cijc_sum_12 = C2_mnimnjWnn_Cijc_sum(1, 2)

# C2(Hninj)*mni*mnj Cijc
def C2_mnimnjHnn_Cijc_sum(a, b):
    return fsum(
    [
        C2_Hnn_caso(b)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

C2_mnimnjHnn_Cijc_sum_23 = C2_mnimnjHnn_Cijc_sum(2, 3)
C2_mnimnjHnn_Cijc_sum_13 = C2_mnimnjHnn_Cijc_sum(1, 3)
C2_mnimnjHnn_Cijc_sum_12 = C2_mnimnjHnn_Cijc_sum(1, 2)

# (mni^2 + mnj^2) C2(Wninj)
def mni2_mnj2_C2_Wnn_Cijc_sum(a, b):
    return fsum(
    [
        (mn[i-1]**2 + mn[j-1]**2)*C2_Wnn_caso(b)[i-1,j-1]*Cij(i-1, j-1)*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

mni2_mnj2_C2_Wnn_Cijc_sum_23 = mni2_mnj2_C2_Wnn_Cijc_sum(2, 3)
mni2_mnj2_C2_Wnn_Cijc_sum_13 = mni2_mnj2_C2_Wnn_Cijc_sum(1, 3)
mni2_mnj2_C2_Wnn_Cijc_sum_12 = mni2_mnj2_C2_Wnn_Cijc_sum(1, 2)


# (mni^2 + mnj^2) C2(Wninj)*mni*mnj
def mni2_mnj2_C2_mnimnjWnn_Cijc_sum(a, b):
    return fsum(
    [
        (mn[i-1]**2 + mn[j-1]**2)*C2_Wnn_caso(b)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

mni2_mnj2_C2_mnimnjWnn_Cijc_sum_23 = mni2_mnj2_C2_mnimnjWnn_Cijc_sum(2, 3)
mni2_mnj2_C2_mnimnjWnn_Cijc_sum_13 = mni2_mnj2_C2_mnimnjWnn_Cijc_sum(1, 3)
mni2_mnj2_C2_mnimnjWnn_Cijc_sum_12 = mni2_mnj2_C2_mnimnjWnn_Cijc_sum(1, 2)

# (mni^2 + mnj^2) C2(Hninj)*mni*mnj
def mni2_mnj2_C2_mnimnjHnn_Cijc_sum(a, b):
    return fsum(
    [
        (mn[i-1]**2 + mn[j-1]**2)*C2_Hnn_caso(b)[i-1,j-1]*mn[i-1]*mn[j-1]*conj(Cij(i-1, j-1))*Unu[b-1,j-1]*Unu_dagger[i-1,a-1] for i in range(1, 10) for j in range(1, 10)
        ]
    )

mni2_mnj2_C2_mnimnjHnn_Cijc_sum_23 = mni2_mnj2_C2_mnimnjHnn_Cijc_sum(2, 3)
mni2_mnj2_C2_mnimnjHnn_Cijc_sum_13 = mni2_mnj2_C2_mnimnjHnn_Cijc_sum(1, 3)
mni2_mnj2_C2_mnimnjHnn_Cijc_sum_12 = mni2_mnj2_C2_mnimnjHnn_Cijc_sum(1, 2)

######################################3
# PV matrix ab
#####################################
# a=2; b=3
# C1_mnimnjWnn_Cijc_sum_ij_2 = C1_mnimnjWnn_Cijc_sum(2, 3)
# C2_mnimnjWnn_Cijc_sum_ij_3 = C2_mnimnjWnn_Cijc_sum(2, 3)
# C2_mni2Hnn_Cij_sum_ij_3 = C2_mni2Hnn_Cij_sum(2, 3)
# C2_mnj2Hnn_Cij_sum_ij_3 = C2_mnj2Hnn_Cij_sum(2, 3)
# C1_mnj2Hnn_Cij_sum_ij_2 = C1_mnj2Hnn_Cij_sum(2, 3)
# C1_mni2Hnn_Cij_sum_ij_2 = C1_mni2Hnn_Cij_sum(2, 3)
# C2_mnimnjHnn_Cijc_sum_ij_3 = C2_mnimnjHnn_Cijc_sum(2, 3)
# C1_mnimnjHnn_Cijc_sum_ij_2 = C1_mnimnjHnn_Cijc_sum(2, 3)
################
# C1_mnj2Wnn_Cij_sum_ij_2 = C1_mnj2Wnn_Cij_sum(2,3)
# C1_mni2Wnn_Cij_sum_ij_2 = C1_mni2Wnn_Cij_sum(2, 3)
# C1_mni2mnj2Wnn_Cij_sum_ij_2 = C1_mni2mnj2Wnn_Cij_sum(2, 3)
# mni2_mnj2_C1_mnimnjWnn_Cijc_sum_ij_2 = mni2_mnj2_C1_mnimnjWnn_Cijc_sum(2, 3)
# C2_mnj2Wnn_Cij_sum_ij_3 = C2_mnj2Wnn_Cij_sum(2, 3)
# C2_mni2Wnn_Cij_sum_ij_3 = C2_mni2Wnn_Cij_sum(2, 3)
# C2_mni2mnj2Wnn_Cij_sum_ij_3 = C2_mni2mnj2Wnn_Cij_sum(2, 3)
# mni2_mnj2_C2_mnimnjWnn_Cijc_sum_ij_3 = mni2_mnj2_C2_mnimnjWnn_Cijc_sum(2, 3)
# mni2_mnj2_C1_Wnn_Cijc_sum_ij_2 = mni2_mnj2_C1_Wnn_Cijc_sum(2, 3)
# mni2_mnj2_C2_Wnn_Cijc_sum_ij_3 = mni2_mnj2_C2_Wnn_Cijc_sum(2, 3)
# C1_mni2mnj2Hnn_Cij_sum_ij_2 = C1_mni2mnj2Hnn_Cij_sum(2, 3)
# mni2_mnj2_C1_mnimnjHnn_Cijc_sum_ij_2 = mni2_mnj2_C1_mnimnjHnn_Cijc_sum(2, 3)
# C2_mni2mnj2Hnn_Cij_sum_ij_3 = C2_mni2mnj2Hnn_Cij_sum(2,3)
# mni2_mnj2_C2_mnimnjHnn_Cijc_sum_ij_3 = mni2_mnj2_C2_mnimnjHnn_Cijc_sum(2, 3)

##################################
######## Factores de forma Xninnj
##################################

from modelos_2HDM import coeff_typeI_h as coeff_h
from modelos_2HDM import tb, cab
from sympy import lambdify
#mA = symbols('m_A',positive=True)
#Kphi =  4*mA**2 - 3*mϕ**2- 2*mHpm**2
#Qphi = mϕ**2 - 2*mHpm**2

xi_nphi = lambdify([tb, cab], coeff_h.xi_nphi, 'mpmath')
xi_lA = lambdify([tb, cab], coeff_h.xi_lA, 'mpmath')
xi_nA = lambdify([tb, cab], coeff_h.xi_nA, 'mpmath')


v = mpf('246')
g = 2*mW/v
F = g**3/(64*pi**2*mW**3)
tb0 = mpf('1e-2')
cab0 = mpf('1e-2')

def form_factors23(tb=tb0, cab=cab0):

    mla = ml[2]
    mlb = ml[3]
    #################3
    ##################
    
    C1_mnimnjWnn_Cijc_sum_ij_a = C1_mnimnjWnn_Cijc_sum_23
    C2_mnimnjWnn_Cijc_sum_ij_b = C2_mnimnjWnn_Cijc_sum_23
    C2_mni2Hnn_Cij_sum_ij_b = C2_mni2Hnn_Cij_sum_23
    C2_mnj2Hnn_Cij_sum_ij_b = C2_mnj2Hnn_Cij_sum_23
    C1_mnj2Hnn_Cij_sum_ij_a = C1_mnj2Hnn_Cij_sum_23
    C1_mni2Hnn_Cij_sum_ij_a = C1_mni2Hnn_Cij_sum_23
    C2_mnimnjHnn_Cijc_sum_ij_b = C2_mnimnjHnn_Cijc_sum_23
    C1_mnimnjHnn_Cijc_sum_ij_a = C1_mnimnjHnn_Cijc_sum_23
    ################
    C1_mnj2Wnn_Cij_sum_ij_a = C1_mnj2Wnn_Cij_sum_23
    C1_mni2Wnn_Cij_sum_ij_a = C1_mni2Wnn_Cij_sum_23
    C1_mni2mnj2Wnn_Cij_sum_ij_a = C1_mni2mnj2Wnn_Cij_sum_23
    mni2_mnj2_C1_mnimnjWnn_Cijc_sum_ij_a = mni2_mnj2_C1_mnimnjWnn_Cijc_sum_23
    C2_mnj2Wnn_Cij_sum_ij_b = C2_mnj2Wnn_Cij_sum_23
    C2_mni2Wnn_Cij_sum_ij_b = C2_mni2Wnn_Cij_sum_23
    C2_mni2mnj2Wnn_Cij_sum_ij_b = C2_mni2mnj2Wnn_Cij_sum_23
    mni2_mnj2_C2_mnimnjWnn_Cijc_sum_ij_b = mni2_mnj2_C2_mnimnjWnn_Cijc_sum_23
    mni2_mnj2_C1_Wnn_Cijc_sum_ij_a = mni2_mnj2_C1_Wnn_Cijc_sum_23
    mni2_mnj2_C2_Wnn_Cijc_sum_ij_b = mni2_mnj2_C2_Wnn_Cijc_sum_23
    C1_mni2mnj2Hnn_Cij_sum_ij_a = C1_mni2mnj2Hnn_Cij_sum_23
    mni2_mnj2_C1_mnimnjHnn_Cijc_sum_ij_a = mni2_mnj2_C1_mnimnjHnn_Cijc_sum_23
    C2_mni2mnj2Hnn_Cij_sum_ij_b = C2_mni2mnj2Hnn_Cij_sum_23
    mni2_mnj2_C2_mnimnjHnn_Cijc_sum_ij_b = mni2_mnj2_C2_mnimnjHnn_Cijc_sum_23
    ############################ 
    B012_mni2_Cij_sum_ij = B012_mni2_Cij_sum_23
    B012_mnj2_Cij_sum_ij = B012_mni2_Cij_sum_23
    B012_mnimnj_Cijc_sum_ij = B012_mnimnj_Cijc_sum_23
    C0_mni2Wnn_Cij_sum_ij = C0_mni2Wnn_Cij_sum_23
    C0_mni2Hnn_Cij_sum_ij = C0_mni2Hnn_Cij_sum_23
    C0_mnj2Wnn_Cij_sum_ij = C0_mnj2Wnn_Cij_sum_23
    C0_mnj2Hnn_Cij_sum_ij = C0_mnj2Hnn_Cij_sum_23
    C0_mnimnjWnn_Cijc_sum_ij = C0_mnimnjWnn_Cijc_sum_23
    C0_mnimnjHnn_Cijc_sum_ij = C0_mnimnjHnn_Cijc_sum_23
    C0_mni2mnj2Hnn_Cij_sum_ij = C0_mni2mnj2Hnn_Cij_sum_23
    C0_mnimnj3Hnn_Cijc_sum_ij = C0_mnimnj3Hnn_Cijc_sum_23
    C0_mni3mnjHnn_Cijc_sum_ij = C0_mni3mnjHnn_Cijc_sum_23

    xi_nphi0 = xi_nphi(tb, cab)
    xi_lA0 = xi_lA(tb, cab)
    xi_nA0 = xi_nA(tb, cab)

    ALGninj = mla*xi_nphi0*F*(
        B012_mnj2_Cij_sum_ij + mW**2*C0_mnj2Wnn_Cij_sum_ij -
        (
            mla**2*C1_mnj2Wnn_Cij_sum_ij_a + mlb**2*C1_mni2Wnn_Cij_sum_ij_a + 2*C1_mni2mnj2Wnn_Cij_sum_ij_a
        ) +
        B012_mnimnj_Cijc_sum_ij + mW**2*C0_mnimnjWnn_Cijc_sum_ij - 
        (mla**2 + mlb**2)*C1_mnimnjWnn_Cijc_sum_ij_a + 
        mni2_mnj2_C1_mnimnjWnn_Cijc_sum_ij_a
    )

    ARGninj = mlb*xi_nphi0*F*(
        B012_mni2_Cij_sum_ij + mW**2*C0_mni2Wnn_Cij_sum_ij +
        (
            mla**2*C2_mnj2Wnn_Cij_sum_ij_b + mlb**2*C2_mni2Wnn_Cij_sum_ij_b - 2*C2_mni2mnj2Wnn_Cij_sum_ij_b
        ) +
        B012_mnimnj_Cijc_sum_ij + mW**2*C0_mnimnjWnn_Cijc_sum_ij +
        (mla**2 + mlb**2)*C2_mnimnjWnn_Cijc_sum_ij_b - 
        mni2_mnj2_C2_mnimnjWnn_Cijc_sum_ij_b
    )

    ALWninj = 2*mW**2*mla*xi_nphi0*F*(
        mni2_mnj2_C1_Wnn_Cijc_sum_ij_a - C0_mnj2Wnn_Cij_sum_ij - C0_mnimnjWnn_Cijc_sum_ij + 2*C1_mnimnjWnn_Cijc_sum_ij_a
    )

    ARWninj = - 2*mW**2*mlb*xi_nphi0*F*(
        mni2_mnj2_C2_Wnn_Cijc_sum_ij_b + C0_mni2Wnn_Cij_sum_ij + C0_mnimnjWnn_Cijc_sum_ij + 2*C2_mnimnjWnn_Cijc_sum_ij_b
    )

    ALHninj = mla*xi_nphi0*F*(
        xi_lA0*xi_nA0*B012_mnj2_Cij_sum_ij + mlb**2*xi_lA0*(xi_lA0 + xi_nA0)*(C2_mni2Hnn_Cij_sum_ij_b + C2_mnj2Hnn_Cij_sum_ij_b) -
        xi_nA0*xi_lA0*mla**2*C1_mnj2Hnn_Cij_sum_ij_a - xi_nA0*xi_lA0*mlb**2*C1_mni2Hnn_Cij_sum_ij_a -
        2*xi_nA0**2*C1_mni2mnj2Hnn_Cij_sum_ij_a + mlb**2*(xi_lA0**2 + xi_lA0*xi_nA0)*C0_mni2Hnn_Cij_sum_ij +
        xi_lA0*xi_nA0*mHpm**2*C0_mnj2Hnn_Cij_sum_ij + (xi_lA0*xi_nA0 + xi_nA0**2)*C0_mni2mnj2Hnn_Cij_sum_ij +
        #
        xi_lA0*xi_nA0*B012_mnimnj_Cijc_sum_ij + 2*xi_lA0*(xi_lA0 + xi_nA0)*mlb**2*C2_mnimnjHnn_Cijc_sum_ij_b -
        xi_lA0*xi_nA0*(mla**2 + mlb**2)*C1_mnimnjHnn_Cijc_sum_ij_a - xi_nA0**2*mni2_mnj2_C1_mnimnjHnn_Cijc_sum_ij_a +
        mlb**2*(xi_lA0**2 + xi_lA0*xi_nA0)*C0_mnimnjHnn_Cijc_sum_ij +
        xi_lA0*xi_nA0*mHpm**2*C0_mnimnjHnn_Cijc_sum_ij +
        (xi_lA0*xi_nA0 + xi_nA0**2)*C0_mnimnj3Hnn_Cijc_sum_ij
        )

    ARHninj = mlb*xi_nphi0*F*(
        xi_lA0*xi_nA0*B012_mni2_Cij_sum_ij - mla**2*xi_lA0*(xi_lA0 + xi_nA0)*(C1_mni2Hnn_Cij_sum_ij_a + C1_mnj2Hnn_Cij_sum_ij_a) -
        xi_nA0*xi_lA0*mla**2*C2_mnj2Hnn_Cij_sum_ij_b + xi_nA0*xi_lA0*mlb**2*C2_mni2Hnn_Cij_sum_ij_b +
        2*xi_nA0**2*C2_mni2mnj2Hnn_Cij_sum_ij_b + mla**2*(xi_lA0**2 + xi_lA0*xi_nA0)*C0_mnj2Hnn_Cij_sum_ij +
        xi_lA0*xi_nA0*mHpm**2*C0_mni2Hnn_Cij_sum_ij + (xi_lA0*xi_nA0 + xi_nA0**2)*C0_mni2mnj2Hnn_Cij_sum_ij +
        #
        xi_lA0*xi_nA0*B012_mnimnj_Cijc_sum_ij - 2*xi_lA0*(xi_lA0 + xi_nA0)*mla**2*C1_mnimnjHnn_Cijc_sum_ij_a -
        xi_lA0*xi_nA0*(mla**2 + mlb**2)*C2_mnimnjHnn_Cijc_sum_ij_b + xi_nA0**2*mni2_mnj2_C2_mnimnjHnn_Cijc_sum_ij_b +
        mla**2*(xi_lA0**2 + xi_lA0*xi_nA0)*C0_mnimnjHnn_Cijc_sum_ij +
        xi_lA0*xi_nA0*mHpm**2*C0_mnimnjHnn_Cijc_sum_ij +
        (xi_lA0*xi_nA0 + xi_nA0**2)*C0_mni3mnjHnn_Cijc_sum_ij
        )
    AL = ALGninj + ALWninj + ALHninj
    AR = ARGninj + ARWninj + ARHninj
    return AL, AR

if __name__ == '__main__':
    def pretty(x):
        nprint(chop(x))
    
    # print('C0(mW, mni, mnj)mni mnj Cijc = ')
    # pretty(C0_mnimnjWnn_Cijc_sum_ij)

    # print('C0(mW, mni, mnj)mnj^2 Cij = ')
    # pretty(C0_mnj2Wnn_Cij_sum_ij)
 
    # print('ALGninj: ')
    # pretty(ALGninj)

    # print('ARGninj: ')
    # pretty(ARGninj)

    # print('ALWninj: ')
    # pretty(ALWninj)

    # print('ARWninj: ')
    # pretty(ARWninj)

    # print('ALHninj: ')
    # pretty(ALHninj)

    # print('ARHninj: ')
    # pretty(ARHninj)
    from time import perf_counter

    start = perf_counter()
    AL, AR = form_factors23()
    end = perf_counter()
    
    print('EL tiempo de ejecución es: \n')
    print((end - start)/60**2)


    print('ALGninj + ALWninj + ALWninj: ')
    pretty(AL)

    print('ARGninj + ARWninj + ARWninj: ')
    pretty(AR)

    #tb_mp = linspae