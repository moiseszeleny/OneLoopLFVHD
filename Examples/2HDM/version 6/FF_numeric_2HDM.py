### Import libraries
from sympy import init_printing,lambdify,Symbol, symbols, Matrix
import OneLoopLFVHD as lfvhd
from FF_symbolic import DiagramasWninj,DiagramasniWW,DiagramasniWH,DiagramasniHW
from FF_symbolic import DiagramasHninj,DiagramasniHH
from FF_symbolic import g, mW, mG, mHpm, mϕ, Uν, Uνc, mn, m, C, Cc, a,b,i
from FF_symbolic import j as jj
from FF_symbolic import ξlϕ, ξnϕ, ξlA, ξnA,α, β, Ξϕ, Kϕ, Qϕ,ρϕ, Δϕ,ηϕ, λ5

from mpmath import *
mp.dps = 80; mp.pretty = True

import numpy as np
import subprocess as s
from multiprocessing import Pool
from time import time

def speedup_array(f,array,procs=4): 
    pool = Pool(procs,maxtasksperchild=100).map(f, array)
    result = np.array(list(pool))
    return result

### Neutrino benchmark
from OneLoopLFVHD.neutrinos import NuOscObservables
Nudata = NuOscObservables

m1 = mpf('1e-12')  #GeV 

#current values to Square mass differences
d21 = mpf(str(Nudata.squareDm21.central))*mpf('1e-18')# factor to convert eV^2 to GeV^2
d31 = mpf(str(Nudata.squareDm31.central))*mpf('1e-18')

#d21 = 7.5e-5*1e-18
#d31 = 2.457e-3*1e-18
m2 = sqrt(m1**2 + d21)
m3 = sqrt(m1**2 + d31)
#######
m4 = lambda m6: m6/3
m5 = lambda m6: m6/2


### Numeric translation 
from OneLoopLFVHD.data import ml

ma,mb = symbols('m_a,m_b',positive=True)
valoresSM ={mW:mpf('80.379'),g:(2*mpf('80.379'))/mpf('246')}
cambios_lab = lambda mla,mlb:{lfvhd.mi:mla,lfvhd.mj:mlb}



Ubi, Ucai,mni = symbols('U_{bi}, {{U_{ai}^*}},m_{n_i}')
UnuOne = {mn[i]:mni,Uν[b,i]:Ubi,Uνc[a,i]:Ucai}

# from Unu_seesaw_2HDM import diagonalizationMnu
# diagonalizationMnu1 = lambda m1,m6,tb,doblet: diagonalizationMnu(
#     m1,m2,m3,m6/mpf('3.0'),m6/mpf('2.0'),m6,tb,doblet)

from Unu_seesaw import diagonalizationMnu

diagonalizationMnu1 = lambda m1,m6: diagonalizationMnu(
    m1,m2,m3,m6/mpf('3.0'),m6/mpf('2.0'),m6)

## Change PV functions by symbolic PV function (for example C0(M0,M1,M2) --> C0)
C0_sp, C1_sp, C2_sp, B120_sp, B10_sp,B11_sp,B20_sp,B21_sp = symbols('C0, C1, C2, B120, B10,B11,B20,B21')
cambios_pave_sympy = lambda M0,M1,M2: {lfvhd.C0(M0,M1,M2):C0_sp,lfvhd.C1(M0,M1,M2):C1_sp,
                                      lfvhd.C2(M0,M1,M2):C2_sp,lfvhd.B12_0(M1,M2):B120_sp,
                                      lfvhd.B1_0(M0,M1):B10_sp,lfvhd.B1_1(M0,M1):B11_sp,
                                      lfvhd.B2_0(M0,M2):B20_sp,lfvhd.B2_1(M0,M2):B21_sp}
cambiosniWW = cambios_pave_sympy(mn[i],mW,mW)
cambiosniWH = cambios_pave_sympy(mn[i],mW,mHpm)
cambiosniHW = cambios_pave_sympy(mn[i],mHpm,mW)
cambiosniHH = cambios_pave_sympy(mn[i],mHpm,mHpm)

cambiosWninj = cambios_pave_sympy(mW,mn[i],mn[jj])
cambiosHninj = cambios_pave_sympy(mHpm,mn[i],mn[jj])
#################################################
### Sum over diagrams with M0 = mni, M1 = M2 = mW
#################################################

import OneLoopLFVHD.LFVHDFeynG_mpmath2 as lfvhd_mp
mW_val = valoresSM[mW]
# print(lfvhd_mp.C0(mpf('125.1'),mpf('1e-12'),mW_val,mW_val))

C0_mp_niWW = lambda ms,mni_: lfvhd_mp.C0(ms,mni_,mW_val,mW_val)
C1_mp_niWW = lambda ms,mni_,mla: lfvhd_mp.C1(ms,mla,mni_,mW_val,mW_val)
C2_mp_niWW = lambda ms,mni_,mlb: lfvhd_mp.C2(ms,mlb,mni_,mW_val,mW_val)
B120_mp_niWW = lambda ms: lfvhd_mp.B12_0(ms,mW_val,mW_val)
B10_mp_niWW = lambda mni_,mla: lfvhd_mp.B1_0(mla,mni_,mW_val)
B11_mp_niWW = lambda mni_,mla: lfvhd_mp.B1_1(mla,mni_,mW_val)
B20_mp_niWW = lambda mni_,mlb: lfvhd_mp.B2_0(mlb,mni_,mW_val)
B21_mp_niWW = lambda mni_,mlb: lfvhd_mp.B2_1(mlb,mni_,mW_val)

FF_list_niWW = []
for dia_niWW in DiagramasniWW:
    A_sp_niWWL = lambda mla,mlb: dia_niWW.AL().subs(cambiosniWW).subs(lfvhd.D,4).subs(
            lfvhd.B12_0(mW,mW),0).simplify().subs(
        cambios_lab(mla,mlb)).subs(valoresSM).subs(UnuOne)
    A_sp_niWWR = lambda mla,mlb: dia_niWW.AR().subs(cambiosniWW).subs(lfvhd.D,4).subs(
            lfvhd.B12_0(mW,mW),0).simplify().subs(
        cambios_lab(mla,mlb)).subs(valoresSM).subs(UnuOne)
    
    
    #display(A_sp_niWWL(ml[1],ml[2]).atoms(Symbol))
    A_lamb_niWWL = lambda mla,mlb: lambdify([lfvhd.ma,mni,Ubi,Ucai, Ξϕ,ξlϕ,
                   C0_sp,C1_sp,C2_sp,B120_sp,B10_sp,B11_sp,B20_sp,B21_sp],
                                           A_sp_niWWL(mla,mlb),'mpmath')
    
    A_lamb_niWWR = lambda mla,mlb: lambdify([lfvhd.ma,mni,Ubi,Ucai, Ξϕ,ξlϕ,
                   C0_sp,C1_sp,C2_sp,B120_sp,B10_sp,B11_sp,B20_sp,B21_sp],
                                           A_sp_niWWR(mla,mlb),'mpmath')
    
    
    #.subs(typeI_ξh).subs(common_factor_h).subs(lfvhd.ma,mh).subs(mh,mpf('125.10'))
    def FFniWWL(ms_val,mla,mlb,mni_,Ubi_,Ucai_,Xi_phi,xi_lphi):
        return A_lamb_niWWL(mla,mlb)(ms_val,mni_,Ubi_,Ucai_,Xi_phi,xi_lphi,
                C0_mp_niWW(ms_val,mni_),C1_mp_niWW(ms_val,mni_,mla),C2_mp_niWW(ms_val,mni_,mlb),
                B120_mp_niWW(ms_val),B10_mp_niWW(mni_,mla),B11_mp_niWW(mni_,mla),
                 B20_mp_niWW(mni_,mlb),B21_mp_niWW(mni_,mlb))
    
    def FFniWWR(ms_val,mla,mlb,mni_,Ubi_,Ucai_,Xi_phi,xi_lphi):
        return A_lamb_niWWR(mla,mlb)(ms_val,mni_,Ubi_,Ucai_,Xi_phi,xi_lphi,
                C0_mp_niWW(ms_val,mni_),C1_mp_niWW(ms_val,mni_,mla),C2_mp_niWW(ms_val,mni_,mlb),
                B120_mp_niWW(ms_val),B10_mp_niWW(mni_,mla),B11_mp_niWW(mni_,mla),
                 B20_mp_niWW(mni_,mlb),B21_mp_niWW(mni_,mlb))
    
    
    FF_list_niWW.append({'L':FFniWWL,'R':FFniWWR})
    
#################################################
### Sum over diagrams with M0 = mni, M1 = M2 = mHpm
#################################################
C0_mp_niHH = lambda ms,mni_,mHpm: lfvhd_mp.C0(ms,mni_,mHpm,mHpm)
C1_mp_niHH = lambda ms,mni_,mla,mHpm: lfvhd_mp.C1(ms,mla,mni_,mHpm,mHpm)
C2_mp_niHH = lambda ms,mni_,mlb,mHpm: lfvhd_mp.C2(ms,mlb,mni_,mHpm,mHpm)
B10_mp_niHH = lambda mni_,mla,mHpm: lfvhd_mp.B1_0(mla,mni_,mHpm)
B11_mp_niHH = lambda mni_,mla,mHpm: lfvhd_mp.B1_1(mla,mni_,mHpm)
B20_mp_niHH = lambda mni_,mlb,mHpm: lfvhd_mp.B2_0(mlb,mni_,mHpm)
B21_mp_niHH = lambda mni_,mlb,mHpm: lfvhd_mp.B2_1(mlb,mni_,mHpm)

FF_list_niHH = []
for dia_niHH in DiagramasniHH:
    A_sp_niHHL = lambda mla,mlb: dia_niHH.AL().subs(cambiosniHH).subs(lfvhd.D,4).simplify().subs(
        cambios_lab(mla,mlb)).subs(valoresSM).subs(UnuOne)
    A_sp_niHHR = lambda mla,mlb: dia_niHH.AR().subs(cambiosniHH).subs(lfvhd.D,4).simplify().subs(
        cambios_lab(mla,mlb)).subs(valoresSM).subs(UnuOne)
    
    
    #display(A_sp_niHHL(ml[1],ml[2]).atoms(Symbol))
    A_lamb_niHHL = lambda mla,mlb: lambdify([lfvhd.ma,mni,Ubi,Ucai,mHpm,β,λ5,ξlϕ,ξlA,ξnA,Δφ,Kφ,Qφ,ρφ,
                   C0_sp,C1_sp,C2_sp,B10_sp,B11_sp,B20_sp,B21_sp],
                                           A_sp_niHHL(mla,mlb),'mpmath')
    
    A_lamb_niHHR = lambda mla,mlb: lambdify([lfvhd.ma,mni,Ubi,Ucai,mHpm,β,λ5,ξlϕ,ξlA,ξnA,Δφ,Kφ,Qφ,ρφ,
                   C0_sp,C1_sp,C2_sp,B10_sp,B11_sp,B20_sp,B21_sp],
                                           A_sp_niHHR(mla,mlb),'mpmath')
    #.subs(typeI_ξh).subs(common_factor_h).subs(lfvhd.ma,mh).subs(mh,mpf('125.10'))
    def FFniHHL(ms_val,mla,mlb,mni_,Ubi_,Ucai_,
           mHpm,beta,l5,xi_lphi,xi_lA,xi_nA,Dphi,Kphi,Qphi,rhophi):
        return A_lamb_niHHL(mla,mlb)(ms_val,mni_,Ubi_,Ucai_,
                      mHpm,beta,l5,xi_lphi,xi_lA,xi_nA,Dphi,Kphi,Qphi,rhophi,
                C0_mp_niHH(ms_val,mni_,mHpm),C1_mp_niHH(ms_val,mni_,mla,mHpm),
                C2_mp_niHH(ms_val,mni_,mlb,mHpm),
                B10_mp_niHH(mni_,mla,mHpm),B11_mp_niHH(mni_,mla,mHpm),
                 B20_mp_niHH(mni_,mlb,mHpm),B21_mp_niHH(mni_,mlb,mHpm))
    
    def FFniHHR(ms_val,mla,mlb,mni_,Ubi_,Ucai_,
           mHpm,beta,l5,xi_lphi,xi_lA,xi_nA,Dphi,Kphi,Qphi,rhophi):
        return A_lamb_niHHR(mla,mlb)(ms_val,mni_,Ubi_,Ucai_,
                      mHpm,beta,l5,xi_lphi,xi_lA,xi_nA,Dphi,Kphi,Qphi,rhophi,
                C0_mp_niHH(ms_val,mni_,mHpm),C1_mp_niHH(ms_val,mni_,mla,mHpm),
                C2_mp_niHH(ms_val,mni_,mlb,mHpm),
                B10_mp_niHH(mni_,mla,mHpm),B11_mp_niHH(mni_,mla,mHpm),
                 B20_mp_niHH(mni_,mlb,mHpm),B21_mp_niHH(mni_,mlb,mHpm))
    
    FF_list_niHH.append({'L':FFniHHL,'R':FFniHHR})
    
#################################################
### Sum over diagrams with M0 = mni, M1 = mW, M2 = mHpm
#################################################
C0_mp_niWH = lambda ms,mni_,mHpm: lfvhd_mp.C0(ms,mni_,mW_val,mHpm)
C1_mp_niWH = lambda ms,mni_,mla,mHpm: lfvhd_mp.C1(ms,mla,mni_,mW_val,mHpm)
C2_mp_niWH = lambda ms,mni_,mlb,mHpm: lfvhd_mp.C2(ms,mlb,mni_,mW_val,mHpm)

FF_list_niWH = []
for dia_niWH in DiagramasniWH:
    A_sp_niWHL = lambda mla,mlb: dia_niWH.AL().subs(
    lfvhd.B12_0(mW,mHpm),0).subs(cambiosniWH).subs(lfvhd.D,4).simplify().subs(
        cambios_lab(mla,mlb)).subs(valoresSM).subs(UnuOne)
    A_sp_niWHR = lambda mla,mlb: dia_niWH.AR().subs(
    lfvhd.B12_0(mW,mHpm),0).subs(cambiosniWH).subs(lfvhd.D,4).simplify().subs(
        cambios_lab(mla,mlb)).subs(valoresSM).subs(UnuOne)
    
    
    #display(A_sp_niWHR(ml[1],ml[2]))
    A_lamb_niWHL = lambda mla,mlb: lambdify([lfvhd.ma,mni,Ubi,Ucai,mHpm,ξlA,ξnA,ηφ,
                   C0_sp,C1_sp,C2_sp], A_sp_niWHL(mla,mlb),'mpmath')
    
    A_lamb_niWHR = lambda mla,mlb: lambdify([lfvhd.ma,mni,Ubi,Ucai,mHpm,ξlA,ξnA,ηφ,
                   C0_sp,C1_sp,C2_sp], A_sp_niWHR(mla,mlb),'mpmath')
    
    #.subs(typeI_ξh).subs(common_factor_h).subs(lfvhd.ma,mh).subs(mh,mpf('125.10'))
    def FFniWHL(ms_val,mla,mlb,mni_,Ubi_,Ucai_,mHpm,xi_lA,xi_nA,etaphi):
        return A_lamb_niWHL(mla,mlb)(ms_val,mni_,Ubi_,Ucai_,
                mHpm,xi_lA,xi_nA,etaphi,
                C0_mp_niWH(ms_val,mni_,mHpm),C1_mp_niWH(ms_val,mni_,mla,mHpm),
                C2_mp_niWH(ms_val,mni_,mlb,mHpm))
    
    def FFniWHR(ms_val,mla,mlb,mni_,Ubi_,Ucai_,mHpm,xi_lA,xi_nA,etaphi):
        return A_lamb_niWHR(mla,mlb)(ms_val,mni_,Ubi_,Ucai_,
                mHpm,xi_lA,xi_nA,etaphi,
                C0_mp_niWH(ms_val,mni_,mHpm),C1_mp_niWH(ms_val,mni_,mla,mHpm),
                C2_mp_niWH(ms_val,mni_,mlb,mHpm))
    
    FF_list_niWH.append({'L':FFniWHL,'R':FFniWHR})
    
#################################################
### Sum over diagrams with M0 = mni, M1 = mHpm, M2 = mW
#################################################

C0_mp_niHW = lambda ms,mni_,mHpm: lfvhd_mp.C0(ms,mni_,mHpm,mW_val)
C1_mp_niHW = lambda ms,mni_,mla,mHpm: lfvhd_mp.C1(ms,mla,mni_,mHpm,mW_val)
C2_mp_niHW = lambda ms,mni_,mlb,mHpm: lfvhd_mp.C2(ms,mlb,mni_,mHpm,mW_val)

FF_list_niHW = []
for dia_niHW in DiagramasniHW:
    A_sp_niHWL = lambda mla,mlb: dia_niHW.AL().subs(
    lfvhd.B12_0(mHpm,mW),0).subs(cambiosniHW).subs(lfvhd.D,4).simplify().subs(
        cambios_lab(mla,mlb)).subs(valoresSM).subs(UnuOne)
    A_sp_niHWR = lambda mla,mlb: dia_niHW.AR().subs(
    lfvhd.B12_0(mHpm,mW),0).subs(cambiosniHW).subs(lfvhd.D,4).simplify().subs(
        cambios_lab(mla,mlb)).subs(valoresSM).subs(UnuOne)
    
    #display(A_sp_niHWL(ml[1],ml[2]).atoms(Symbol))
    A_lamb_niHWL = lambda mla,mlb: lambdify([lfvhd.ma,mni,Ubi,Ucai,mHpm,ξlA,ξnA,ηφ,
                   C0_sp,C1_sp,C2_sp], A_sp_niHWL(mla,mlb),'mpmath')
    
    A_lamb_niHWR = lambda mla,mlb: lambdify([lfvhd.ma,mni,Ubi,Ucai,mHpm,ξlA,ξnA,ηφ,
                   C0_sp,C1_sp,C2_sp], A_sp_niHWR(mla,mlb),'mpmath')
    
    #.subs(typeI_ξh).subs(common_factor_h).subs(lfvhd.ma,mh).subs(mh,mpf('125.10'))
    def FFniHWL(ms_val,mla,mlb,mni_,Ubi_,Ucai_,mHpm,xi_lA,xi_nA,etaphi):
        return A_lamb_niHWL(mla,mlb)(ms_val,mni_,Ubi_,Ucai_,
                mHpm,xi_lA,xi_nA,etaphi,
                C0_mp_niHW(ms_val,mni_,mHpm),C1_mp_niHW(ms_val,mni_,mla,mHpm),
                C2_mp_niHW(ms_val,mni_,mlb,mHpm))
    
    def FFniHWR(ms_val,mla,mlb,mni_,Ubi_,Ucai_,mHpm,xi_lA,xi_nA,etaphi):
        return A_lamb_niHWR(mla,mlb)(ms_val,mni_,Ubi_,Ucai_,
                mHpm,xi_lA,xi_nA,etaphi,
                C0_mp_niHW(ms_val,mni_,mHpm),C1_mp_niHW(ms_val,mni_,mla,mHpm),
                C2_mp_niHW(ms_val,mni_,mlb,mHpm))
    
    FF_list_niHW.append({'L':FFniHWL,'R':FFniHWR})

## Internal funtions TwoFermion

mnj = symbols('m_{n_j}',positive=True)
Cijs, Cijcs, Ubj = symbols('C_{ij}, {{C_{ij}^*}},U_{bj}')
UnuTwo = {mn[i]:mni,mn[jj]:mnj,C[i,jj]:Cijs, Cc[i,jj]:Cijcs, Uν[b,jj]:Ubj, Uνc[a,i]:Ucai}

#################################################
### Sum over diagrams with M0 = mW, M1 = mni, M2 = mnj
#################################################
C0_mp_Wninj = lambda ms,mni_,mnj_: lfvhd_mp.C0(ms,mW_val,mni_,mnj_)
C1_mp_Wninj = lambda ms,mni_,mnj_,mla: lfvhd_mp.C1(ms,mla,mW_val,mni_,mnj_)
C2_mp_Wninj = lambda ms,mni_,mnj_,mlb: lfvhd_mp.C2(ms,mlb,mW_val,mni_,mnj_)
B120_mp_Wninj = lambda ms,mni_,mnj_: lfvhd_mp.B12_0(ms,mni_,mnj_)

FF_list_Wninj = []#.subs(lfvhd.D,4).subs(cambios_hab(a,b)).subs(valores).subs(UnuTwo)
for dia_Wninj in DiagramasWninj:
    A_sp_WninjL = lambda mla,mlb: dia_Wninj.AL().subs(cambiosWninj).subs(lfvhd.D,4
                    ).simplify().subs(cambios_lab(mla,mlb)).subs(valoresSM).subs(UnuTwo)
    A_sp_WninjR = lambda mla,mlb: dia_Wninj.AR().subs(cambiosWninj).subs(lfvhd.D,4
                    ).simplify().subs(cambios_lab(mla,mlb)).subs(valoresSM).subs(UnuTwo)
    
    
    #display(A_sp_WninjL(ml[1],ml[2]).atoms(Symbol))
    A_lamb_WninjL = lambda mla,mlb: lambdify([lfvhd.ma,mni,mnj,Ubj,Ucai,Cijs,Cijcs,ξnϕ,
                   C0_sp,C1_sp,B120_sp], A_sp_WninjL(mla,mlb),'mpmath')
    
    A_lamb_WninjR = lambda mla,mlb: lambdify([lfvhd.ma,mni,mnj,Ubj,Ucai,Cijs,Cijcs,ξnϕ,
                   C0_sp,C2_sp,B120_sp], A_sp_WninjR(mla,mlb),'mpmath')
    
    
    #.subs(typeI_ξh).subs(common_factor_h).subs(lfvhd.ma,mh).subs(mh,mpf('125.10'))
    def FFWninjL(ms_val,mla,mlb,mni_,mnj_,Ubj_,Ucai_,Cijs_,Cijcs_,xi_nphi):
        return A_lamb_WninjL(mla,mlb)(
            ms_val,mni_,mnj_,Ubj_,Ucai_,Cijs_,Cijcs_,xi_nphi,
            C0_mp_Wninj(ms_val,mni_,mnj_),C1_mp_Wninj(ms_val,mni_,mnj_,mla),
            B120_mp_Wninj(ms_val,mni_,mnj_)
        )
    def FFWninjR(ms_val,mla,mlb,mni_,mnj_,Ubj_,Ucai_,Cijs_,Cijcs_,xi_nphi):
        return A_lamb_WninjR(mla,mlb)(
            ms_val,mni_,mnj_,Ubj_,Ucai_,Cijs_,Cijcs_,xi_nphi,
            C0_mp_Wninj(ms_val,mni_,mnj_),
            C2_mp_Wninj(ms_val,mni_,mnj_,mlb),B120_mp_Wninj(ms_val,mni_,mnj_)
        )
    
    FF_list_Wninj.append({'L':FFWninjL,'R':FFWninjR})#
    

#################################################
### Sum over diagrams with M0 = mHpm, M1 = mni, M2 = mnj
#################################################
C0_mp_Hninj = lambda ms,mni_,mnj_,mHpm: lfvhd_mp.C0(ms,mHpm,mni_,mnj_)
C1_mp_Hninj = lambda ms,mni_,mnj_,mla,mHpm: lfvhd_mp.C1(ms,mla,mHpm,mni_,mnj_)
C2_mp_Hninj = lambda ms,mni_,mnj_,mlb,mHpm: lfvhd_mp.C2(ms,mlb,mHpm,mni_,mnj_)
B120_mp_Hninj = lambda ms,mni_,mnj_: lfvhd_mp.B12_0(ms,mni_,mnj_)

FF_list_Hninj = []#.subs(lfvhd.D,4).subs(cambios_hab(a,b)).subs(valores).subs(UnuTwo)
for dia_Hninj in DiagramasHninj:
    A_sp_HninjL = lambda mla,mlb: dia_Hninj.AL().subs(cambiosHninj).subs(lfvhd.D,4
                    ).simplify().subs(cambios_lab(mla,mlb)).subs(valoresSM).subs(UnuTwo)
    A_sp_HninjR = lambda mla,mlb: dia_Hninj.AR().subs(cambiosHninj).subs(lfvhd.D,4
                    ).simplify().subs(cambios_lab(mla,mlb)).subs(valoresSM).subs(UnuTwo)
    
    
    #display(A_sp_HninjL(ml[1],ml[2]).atoms(Symbol))
    A_lamb_HninjL = lambda mla,mlb: lambdify([lfvhd.ma,mni,mnj,Ubj,Ucai,Cijs,Cijcs,mHpm,
                ξnϕ,ξnA,ξlA,C0_sp,C1_sp,C2_sp,B120_sp], A_sp_HninjL(mla,mlb),'mpmath')
    
    A_lamb_HninjR = lambda mla,mlb: lambdify([lfvhd.ma,mni,mnj,Ubj,Ucai,Cijs,Cijcs,mHpm,
                ξnϕ,ξnA,ξlA,C0_sp,C1_sp,C2_sp,B120_sp], A_sp_HninjR(mla,mlb),'mpmath')
    
    
    #.subs(typeI_ξh).subs(common_factor_h).subs(lfvhd.ma,mh).subs(mh,mpf('125.10'))
    def FFHninjL(ms_val,mla,mlb,mni_,mnj_,Ubj_,Ucai_,Cijs_,Cijcs_,mHpm,
                 xi_nphi,xi_nA,xi_lA):
        return A_lamb_HninjL(mla,mlb)(
            ms_val,mni_,mnj_,Ubj_,Ucai_,Cijs_,Cijcs_,mHpm,
            xi_nphi,xi_nA,xi_lA,
            C0_mp_Hninj(ms_val,mni_,mnj_,mHpm),C1_mp_Hninj(ms_val,mni_,mnj_,mla,mHpm),
            C2_mp_Hninj(ms_val,mni_,mnj_,mlb,mHpm),B120_mp_Hninj(ms_val,mni_,mnj_)
        )
    def FFHninjR(ms_val,mla,mlb,mni_,mnj_,Ubj_,Ucai_,Cijs_,Cijcs_,mHpm,
                 xi_nphi,xi_nA,xi_lA):
        return A_lamb_HninjR(mla,mlb)(
            ms_val,mni_,mnj_,Ubj_,Ucai_,Cijs_,Cijcs_,mHpm,
            xi_nphi,xi_nA,xi_lA,
            C0_mp_Hninj(ms_val,mni_,mnj_,mHpm),C1_mp_Hninj(ms_val,mni_,mnj_,mla,mHpm),
            C2_mp_Hninj(ms_val,mni_,mnj_,mlb,mHpm),B120_mp_Hninj(ms_val,mni_,mnj_)
        )
    
    FF_list_Hninj.append({'L':FFHninjL,'R':FFHninjR})#


#######################################################3
# Total sum over neutrino generations
###########################################################
def sum_diagrams(m6,aa,bb,ms_val,mHpm,beta,l5,
                   xi_lphi,xi_lA,xi_nphi,xi_nA,Xi_phi,Dphi,Kphi,Qphi,rhophi,etaphi,quirality='L'):
    
    mnk,UnuL,UnuR = diagonalizationMnu1(m1,m6)
    Unu = UnuL
    Unu_dagger = UnuR
    mla = ml[aa]
    mlb = ml[bb]
    if quirality=='L':
        pass
    elif quirality=='R':
        pass
    else:
        raise ValueError('quirality must be L or R')
    
    FFOne = 0
    for k in range(1,7):
        for FF_dict in FF_list_niWW:
            FFOne += FF_dict[quirality](ms_val,mla,mlb,
            mnk[k-1],Unu[bb-1,k-1],Unu_dagger[k-1,aa-1],
            Xi_phi,xi_lphi)
        
        for FF_dict in FF_list_niHW:
            FFOne += FF_dict[quirality](ms_val,mla,mlb,
            mnk[k-1],Unu[bb-1,k-1],Unu_dagger[k-1,aa-1],
            mHpm,xi_lA,xi_nA,etaphi)
        
        for FF_dict in FF_list_niWH:
            FFOne += FF_dict[quirality](ms_val,mla,mlb,
            mnk[k-1],Unu[bb-1,k-1],Unu_dagger[k-1,aa-1],
            mHpm,xi_lA,xi_nA,etaphi)
            
        for FF_dict in FF_list_niHH:
            FFOne += FF_dict[quirality](ms_val,mla,mlb,
            mnk[k-1],Unu[bb-1,k-1],Unu_dagger[k-1,aa-1],
            mHpm,beta,l5,xi_lphi,xi_lA,xi_nA,Dphi,Kphi,Qphi,rhophi)
                                
        
    FFTwo = 0
    Cij = lambda i,j: mp.fsum([Unu[c,i]*Unu_dagger[j,c] for c in range(3)])

    for p in range(1,7):
        for q in range(1,7):
            for FF_dict in FF_list_Wninj:
                FFTwo += FF_dict[quirality](ms_val,mla,mlb,
                mnk[p-1],mnk[q-1],Unu[bb-1,q-1],Unu_dagger[p-1,aa-1],
                Cij(p-1,q-1),conj(Cij(p-1,q-1)),xi_nphi)
            
            for FF_dict in FF_list_Hninj:
                FFTwo += FF_dict[quirality](ms_val,mla,mlb,
                mnk[p-1],mnk[q-1],
                Unu[bb-1,q-1],Unu_dagger[p-1,aa-1],
                Cij(p-1,q-1),conj(Cij(p-1,q-1)),
                mHpm,xi_nphi,xi_nA,xi_lA)
    
    FFtotal = FFTwo + FFOne
    return FFtotal

from modelos_2HDM import typeI_h, typeII_h, Lepton_specific_h,Flipped_h # Acoplamientos

def numeric_sum_diagrams(ms,a,b,mHpm, mA, alpha, beta, l5,
                            type_2HDM=typeI_h,quirality='L'):
    if quirality=='L':
        pass
    elif quirality=='R':
        pass
    else:
        raise ValueError('quirality must be L or R')
    Kphi =  4*mA**2 - 3*ms**2- 2*mHpm**2
    Qphi = ms**2 - 2*mHpm**2
    
    # type_2HDM = type_2HDM.upper()
    # if type_2HDM == 'I':
    #     Yuk_common = typeI_h(alpha,beta,H_a)
    # elif type_2HDM == 'II':
    #     Yuk_common = typeII_h(alpha,beta,H_a)
    # elif type_2HDM == 'Lepton-specific'.upper():
    #     Yuk_common = Lepton_specific_h(alpha,beta,H_a)
    # elif type_2HDM == 'Flipped'.upper():
    #     Yuk_common = Flipped_h(alpha,beta,H_a)
    # else:
    #     raise ValueError('The possible values for type_2HDM are "I", "II", "Lepton-specific" and "Flipped"')
    
    Yuk_common = type_2HDM(alpha,beta)
    xi_lphi, xi_nphi, xi_lA, xi_nA, Xi_phi, etaphi,rhophi,Dphi = Yuk_common
    def FFOne(m6):
        out = sum_diagrams(m6,a,b,ms,mHpm,beta,l5,
                   xi_lphi,xi_lA,xi_nphi,xi_nA,Xi_phi,Dphi,Kphi,Qphi,rhophi,etaphi,quirality)
        return out
    return FFOne
####################################################################################
####################################################################################
####################################################################################

def ALtot23(ms,m6,mHpm, mA, alpha, beta, l5,type_2HDM=typeI_h):
    return numeric_sum_diagrams(ms,2,3,mHpm, mA, alpha, beta, l5,
                                   type_2HDM=type_2HDM,quirality='L')(m6)
def ARtot23(ms,m6,mHpm, mA, alpha, beta, l5,type_2HDM=typeI_h):
    return numeric_sum_diagrams(ms,2,3,mHpm, mA, alpha, beta, l5,
                                   type_2HDM=type_2HDM,quirality='R')(m6)


def ALtot13(ms,m6,mHpm, mA, alpha, beta, l5,type_2HDM=typeI_h):
    return numeric_sum_diagrams(ms,1,3,mHpm, mA, alpha, beta, l5,
                                   type_2HDM=type_2HDM,quirality='L')(m6)
def ARtot13(ms,m6,mHpm, mA, alpha, beta, l5,type_2HDM=typeI_h):
    return numeric_sum_diagrams(ms,1,3,mHpm, mA, alpha, beta, l5,
                                   type_2HDM=type_2HDM,quirality='R')(m6)


def ALtot12(ms,m6,mHpm, mA, alpha, beta, l5,type_2HDM=typeI_h):
    return numeric_sum_diagrams(ms,1,2,mHpm, mA, alpha, beta, l5,
                                   type_2HDM=type_2HDM,quirality='L')(m6)
def ARtot12(ms,m6,mHpm, mA, alpha, beta, l5,type_2HDM=typeI_h):
    return numeric_sum_diagrams(ms,1,2,mHpm, mA, alpha, beta, l5,
                                   type_2HDM=type_2HDM,quirality='R')(m6)



####################################################################################
####################################################################################
####################################################################################    
## Sum over diagrams with two fermions
# def sum_diagramsTwo(m6,a,b,ms_val,mHpm,xi_nphi,xi_nA,xi_lA,quirality='L'):
#     FFTwo = 0
#     mnk,UnuL,UnuR = diagonalizationMnu1(m1,m6)
#     Unu = UnuL
#     Unu_dagger = UnuR
#     Cij = lambda i,j: mp.fsum([Unu[c,i]*Unu_dagger[j,c] for c in range(3)])
    
#     mla = ml[a]
#     mlb = ml[b]
#     if quirality=='L':
#         pass
#     elif quirality=='R':
#         pass
#     else:
#         raise ValueError('quirality must be L or R')
        
#     for p in range(1,7):
#         for q in range(1,7):
#             for FF_dict in FF_list_Wninj:
#                 FFTwo += FF_dict[quirality](ms_val,mla,mlb,
#                         mnk[p-1],mnk[q-1],Unu[b-1,q-1],Unu_dagger[p-1,a-1],
#                   Cij(p-1,q-1),conj(Cij(p-1,q-1)),xi_nphi)
                
#             for FF_dict in FF_list_Hninj:
#                 FFTwo += FF_dict[quirality](ms_val,mla,mlb,
#                                             mnk[p-1],mnk[q-1],
#                                             Unu[b-1,q-1],Unu_dagger[p-1,a-1],
#                                             Cij(p-1,q-1),conj(Cij(p-1,q-1)),
#                                             mHpm,xi_nphi,xi_nA,xi_lA)
        
#     return FFTwo

# def numeric_sum_diagramsTwo(ms,a,b,mHpm, alpha, beta,
#                             H_a='h',type_2HDM='I',quirality='L'):
#     if quirality=='L':
#         pass
#     elif quirality=='R':
#         pass
#     else:
#         raise ValueError('quirality must be L or R')
    
#     type_2HDM = type_2HDM.upper()
#     if type_2HDM == 'I':
#         Yuk_common = typeI_h(alpha,beta,H_a)
#     elif type_2HDM == 'II':
#         Yuk_common = typeII_h(alpha,beta,H_a)
#     elif type_2HDM == 'Lepton-specific'.upper():
#         Yuk_common = Lepton_specific_h(alpha,beta,H_a)
#     elif type_2HDM == 'Flipped'.upper():
#         Yuk_common = Flipped_h(alpha,beta,H_a)
#     else:
#         raise ValueError('The possible values for type_2HDM are "I", "II", "Lepton-specific" and "Flipped"')
    
#     xi_lphi, xi_nphi, xi_lA, xi_nA, Xi_phi, etaphi,rhophi,Dphi = Yuk_common
#     def FFTwo(m6):
#         out = sum_diagramsTwo(m6,a,b,ms,mHpm,xi_nphi,xi_nA,xi_lA,quirality)
#         return out
#     return FFTwo
####################################################################################
####################################################################################
#################################################################################### 

# def ALTwoTot23(ms,m6,mHpm,alpha,beta,H_a='h',type_2HDM='I'):
#     return numeric_sum_diagramsTwo(ms,2,3,mHpm, alpha, beta,
#                                    H_a=H_a,type_2HDM=type_2HDM,quirality='L')(m6)
# def ARTwoTot23(ms,m6,mHpm,alpha,beta,H_a='h',type_2HDM='I'):
#     return numeric_sum_diagramsTwo(ms,2,3,mHpm, alpha, beta,
#                                    H_a=H_a,type_2HDM=type_2HDM,quirality='R')(m6)

# def ALTwoTot13(ms,m6,mHpm,alpha,beta,H_a='h',type_2HDM='I'):
#     return numeric_sum_diagramsTwo(ms,1,3,mHpm, alpha, beta,
#                                    H_a=H_a,type_2HDM=type_2HDM,quirality='L')(m6)
# def ARTwoTot13(ms,m6,mHpm,alpha,beta,H_a='h',type_2HDM='I'):
#     return numeric_sum_diagramsTwo(ms,1,3,mHpm, alpha, beta,
#                                    H_a=H_a,type_2HDM=type_2HDM,quirality='R')(m6)

# def ALTwoTot12(ms,m6,mHpm,alpha,beta,H_a='h',type_2HDM='I'):
#     return numeric_sum_diagramsTwo(ms,1,2,mHpm, alpha, beta,
#                                    H_a=H_a,type_2HDM=type_2HDM,quirality='L')(m6)
# def ARTwoTot12(ms,m6,mHpm,alpha,beta,H_a='h',type_2HDM='I'):
#     return numeric_sum_diagramsTwo(ms,1,2,mHpm, alpha, beta,
#                                    H_a=H_a,type_2HDM=type_2HDM,quirality='R')(m6)
################3
# Total Form Factor
###################
#a = 2, b = 3
# def ALtot23(ms,m6,mHpm, mA, alpha, beta, l5,H_a='h',type_2HDM='I'):
#     ALOne = ALOneTot23(ms,m6,mHpm, mA, alpha, beta, l5,H_a=H_a,type_2HDM=type_2HDM)
#     ALTwo = ALTwoTot23(ms,m6,mHpm,alpha,beta,H_a=H_a,type_2HDM=type_2HDM)
#     return  ALOne + ALTwo

# def ARtot23(ms,m6,mHpm, mA, alpha, beta, l5,H_a='h',type_2HDM='I'):
#     AROne = AROneTot23(ms,m6,mHpm, mA, alpha, beta, l5,H_a=H_a,type_2HDM=type_2HDM)
#     ARTwo = ARTwoTot23(ms,m6,mHpm,alpha,beta,H_a=H_a,type_2HDM=type_2HDM)
#     return  AROne + ARTwo

# #a = 1, b = 3
# def ALtot13(ms,m6,mHpm, mA, alpha, beta, l5,H_a='h',type_2HDM='I'):
#     ALOne = ALOneTot13(ms,m6,mHpm, mA, alpha, beta, l5,H_a=H_a,type_2HDM=type_2HDM)
#     ALTwo = ALTwoTot13(ms,m6,mHpm,alpha,beta,H_a=H_a,type_2HDM=type_2HDM)
#     return  ALOne + ALTwo

# def ARtot13(ms,m6,mHpm, mA, alpha, beta, l5,H_a='h',type_2HDM='I'):
#     AROne = AROneTot13(ms,m6,mHpm, mA, alpha, beta, l5,H_a=H_a,type_2HDM=type_2HDM)
#     ARTwo = ARTwoTot13(ms,m6,mHpm,alpha,beta,H_a=H_a,type_2HDM=type_2HDM)
#     return  AROne + ARTwo

# #a = 1, b = 2
# def ALtot12(ms,m6,mHpm, mA, alpha, beta, l5,H_a='h',type_2HDM='I'):
#     ALOne = ALOneTot12(ms,m6,mHpm, mA, alpha, beta, l5,H_a=H_a,type_2HDM=type_2HDM)
#     ALTwo = ALTwoTot12(ms,m6,mHpm,alpha,beta,H_a=H_a,type_2HDM=type_2HDM)
#     return  ALOne + ALTwo

# def ARtot12(ms,m6,mHpm, mA, alpha, beta, l5,H_a='h',type_2HDM='I'):
#     AROne = AROneTot12(ms,m6,mHpm, mA, alpha, beta, l5,H_a=H_a,type_2HDM=type_2HDM)
#     ARTwo = ARTwoTot12(ms,m6,mHpm,alpha,beta,H_a=H_a,type_2HDM=type_2HDM)
#     return  AROne + ARTwo

### Def angles alpha and beta
def betaf(tb):
    return mp.atan(tb)
def alphaf(tb,x0=mp.mpf('0.01')): #Alignment h SM-like.
    return mp.atan(tb) - mp.acos(x0)


###############################################
# we set cos(beta - alpha)=0.01 and create the cases
###############################################
Benchmarck1 = {'mA':'800','mHpm':'1000.0','l5':'0.1', 'mn6':'1e10','n':'1'}
Benchmarck2 = {'mA':'800','mHpm':'1000.0','l5':'1', 'mn6':'1e15','n':'2'}
Benchmarck3 = {'mA':'1300','mHpm':'1500.0','l5':'1', 'mn6':'1e15','n':'3'}
Benchmarck4 = {'mA':'1300','mHpm':'1500.0','l5':'1', 'mn6':'1e15','n':'4'}


################################################
### We set com setings of each model to automatize
################################################
modelo_typeI = lambda caso: {'modelo':typeI_h,'carpeta':'Type_I',
                             'filename':f"TypeI_Cab095_caso{caso['n']}"}
modelo_typeII = lambda caso: {'modelo':typeII_h,'carpeta':'Type_II',
                             'filename':f"TypeII_Cab095_caso{caso['n']}"}
modelo_lepton = lambda caso: {'modelo':Lepton_specific_h,'carpeta':'Lepton_specific',
                              'filename':f"Lepton_specific_Cab095_caso{caso['n']}"}
modelo_flipped = lambda caso: {'modelo':Flipped_h,'carpeta':'Flipped',
                              'filename':f"Flipped_Cab095_caso{caso['n']}"}

modelos = [modelo_typeI,modelo_typeII,modelo_lepton ,modelo_flipped]

casos = [Benchmarck1, Benchmarck2, Benchmarck3, Benchmarck4]

mh = mpf('125.1')# SM-like Higgs mass

def ALtot23_caso1(tb,m6,mHpm,mA, l5,type_2HDM=typeI_h):
    return ALtot23(mh,m6,mHpm, mA, alphaf(tb), betaf(tb), l5,type_2HDM=type_2HDM)
def ARtot23_caso1(tb,m6,mHpm,mA, l5,type_2HDM=typeI_h):
    return ARtot23(mh,m6,mHpm, mA, alphaf(tb), betaf(tb), l5,type_2HDM=type_2HDM)

def ALtot13_caso1(tb,m6,mHpm,mA, l5,type_2HDM=typeI_h):
    return ALtot13(mh,m6,mHpm, mA, alphaf(tb), betaf(tb), l5,type_2HDM=type_2HDM)
def ARtot13_caso1(tb,m6,mHpm,mA, l5,type_2HDM=typeI_h):
    return ARtot13(mh,m6,mHpm, mA, alphaf(tb), betaf(tb), l5,type_2HDM=type_2HDM)

def ALtot12_caso1(tb,m6,mHpm,mA, l5,type_2HDM=typeI_h):
    return ALtot12(mh,m6,mHpm, mA, alphaf(tb), betaf(tb), l5,type_2HDM=type_2HDM)
def ARtot12_caso1(tb,m6,mHpm,mA, l5,type_2HDM=typeI_h):
    return ARtot12(mh,m6,mHpm, mA, alphaf(tb), betaf(tb), l5,type_2HDM=type_2HDM)

if __name__ == '__main__':
    #nprint((abs(ALTwoTot12(0.1,1,2,3,typeI_ξh))))
    #nprint(ALtot12(m1,1,2,3,4,5,typeI_ξh)),label=r'Br($h \to e \tau$)'

    
    
    ### Width decay
    from time import perf_counter
    from OneLoopLFVHD import Γhlilj 
    from pandas import DataFrame
    import matplotlib.pyplot as plt
    #n = 3#800
    n = int(input("n = "))
    expmp = linspace(-2,2,n)
    tbmp = np.array([mpf('10.0')**k for k in expmp])#np.logspace(-1,15,n)
    #y1 = np.ones_like(tbmp)
    Atlas_bound13 = 0.00047
    Atlas_bound23 = 0.00028
    #y2 = np.ones_like(tbmp)*Atlas_bound23
    
    start = perf_counter()
    for modelo in [modelo_typeI]:#modelos:
        for caso in casos:

            mHpm_val = mp.mpf(caso['mHpm'])
            mA_val = mp.mpf(caso['mA']) 
            l5_val = mp.mpf(caso['l5']) 
            m6_val = mp.mpf(caso['mn6'])
            ######################################
            ###### here we choose the model
            ######################################

            def Γhl2l3(tb):
                AL = ALtot23_caso1(tb,m6=m6_val,mHpm=mHpm_val,mA=mA_val,l5=l5_val,
                                           type_2HDM=modelo(caso)['modelo'])
                AR = ARtot23_caso1(tb,m6=m6_val,mHpm=mHpm_val,mA=mA_val,l5=l5_val,
                                           type_2HDM=modelo(caso)['modelo'])
                return Γhlilj(AL,AR, mh,ml[2],ml[3])
            
            def Γhl1l3(tb):
                AL = ALtot13_caso1(tb,m6=m6_val,mHpm=mHpm_val,mA=mA_val,l5=l5_val,
                                           type_2HDM=modelo(caso)['modelo'])
                AR = ARtot13_caso1(tb,m6=m6_val,mHpm=mHpm_val,mA=mA_val,l5=l5_val,
                                           type_2HDM=modelo(caso)['modelo'])
                return Γhlilj(AL,AR, mh,ml[1],ml[3])
            
            def Γhl1l2(tb):
                AL = ALtot12_caso1(tb,m6=m6_val,mHpm=mHpm_val,mA=mA_val,l5=l5_val,
                                           type_2HDM=modelo(caso)['modelo'])
                AR = ARtot12_caso1(tb,m6=m6_val,mHpm=mHpm_val,mA=mA_val,l5=l5_val,
                                           type_2HDM=modelo(caso)['modelo'])
                return Γhlilj(AL,AR, mh,ml[1],ml[2])
            


            YW23 = speedup_array(Γhl2l3,tbmp)
            YW13 = speedup_array(Γhl1l3,tbmp)
            YW12 = speedup_array(Γhl1l2,tbmp)
            # YW32 = speedup_array(Γhl3l2,tbmp)
            # YW31 = speedup_array(Γhl3l1,tbmp)
            # YW21 = speedup_array(Γhl2l1,tbmp)

            Wtot = YW23 + YW13 + YW12 + 0.0032
            # Wtot = YW32 + YW31 + YW21 + 0.0032

            ###############
            # Plot
            ###############
            plt.figure(figsize=(15,8))
            plt.loglog(np.real(tbmp),(YW23)/Wtot,label=r'Br($h \to \mu \tau$)')
            plt.loglog(np.real(tbmp),(YW13)/Wtot,label=r'Br($h \to e \tau$)')
            plt.loglog(np.real(tbmp),(YW12)/Wtot,label=r'Br($h \to e \mu$)')
            plt.xlim(1e-2,1e2)
            # Horizontal lines
            plt.hlines(1,0.01,1e2,linestyles='-',color='brown')
            plt.hlines(Atlas_bound23,0.01,1e2,linestyles='-',color='brown')
            plt.axhspan(Atlas_bound23, 1,1e-2,1e2,color='brown',alpha=0.5)
            #plt.fill_between(tbmp,y2,y1,color='brown',alpha=0.5)
            plt.text(1e-1,1e-2,r"Atlas bound $\mathcal{BR}(h \to \mu \tau)$",fontsize=18)
            #plt.hlines(1e-46,0.1,1e2,linestyles='--',color='b',label=r'$1\times 10^{-46}$')

            # Vertical lines
            #plt.vlines(1,1e-46,1e-9,linestyles='--',color='r',label=r'$\tan{\beta}=1$')
            #Axis
            #plt.yticks([1e-49,1e-39,1e-29,1e-19,1e-9],fontsize=18)
            #plt.xticks([0.1,1,10,100],fontsize=18)
            plt.yticks(fontsize=18)
            plt.xticks(fontsize=18)
            plt.xlabel(r'$\tan{\beta}$',fontsize=18)
            plt.ylabel(r'$\mathcal{BR}(h \to e_a e_b)$',fontsize=18)
            #plt.title(r'$m_A=$ GeV, $m_{H^{\pm}}=1000$ GeV, $m_{n_6}={10^{10}}$ GeV,$\lambda_5=0.1$',fontsize=18)
            plt.title(f"$m_A=${caso['mA']} GeV"+ r" $m_{H^{\pm}}=$" + f"{caso['mHpm']} GeV" + r" $m_{n_6}=$" + f"{caso['mn6']} GeV" + r" $\lambda_5=$" + f"{caso['l5']}" ,fontsize=18)
            plt.legend(fontsize=18,frameon=True,ncol=1)

            path = f"output/{modelo(caso)['carpeta']}/"
            plt.savefig(path+f"{modelo(caso)['filename']}.png",dpi=100)
            #plt.show()

            ###############
            # Pandas export
            ###############
            df = DataFrame({'tb':tbmp,
                           'Whl2l3':YW23,
                           #'Whl3l2':YW32,
                           'Whl1l3':YW13,
                           #'Whl3l1':YW31,
                           'Whl1l2':YW12})
                           #'Whl2l1':YW21})

            df.to_csv(path+f"{modelo(caso)['filename']}.txt",sep='\t')
            print(path+f"{modelo(caso)['filename']}.txt has been completed")
    plt.show()
    
    end = perf_counter()
    
    print('EL tiempo de ejecución es: \n')
    print(end - start)

    
