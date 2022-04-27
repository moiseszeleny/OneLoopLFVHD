### Import libraries
from sympy import init_printing,lambdify,Symbol, symbols, Matrix
import OneLoopLFVHD as lfvhd
from THDM_seesaw_FF import TrianglesOneFermion, TrianglesTwoFermion, Bubbles
from THDM_seesaw_FF import g, mW, mG, mHpm, mϕ, Uν, Uνc, mn, m, C, Cc, a,b,i
from THDM_seesaw_FF import j as jj
from THDM_seesaw_FF import ξlϕ, ξnϕ, ξlA, ξnA,α, β, Ξϕ, Kϕ, Qϕ,ρϕ, Δϕ,ηϕ, λ5

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

# Settings 2HDM
from sympy import cos as cos_sp
from sympy import sin as sin_sp
from sympy import cot as cot_sp
from sympy import tan as tan_sp
## 2HDMs ###### Cambie los signos de ξnA #########
typeI_ξh = {ξlφ:cos_sp(α)/sin_sp(β),ξnφ:cos_sp(α)/sin_sp(β),ξlA:-cot_sp(β),ξnA:cot_sp(β)}
#typeI_ξH = {ξlφ:sin_sp(α)/sin_sp(β),ξnφ:sin_sp(α)/sin_sp(β),ξlA:-cot_sp(β),ξnA:cot_sp(β)}

typeII_ξh = {ξlφ:-sin_sp(α)/cos_sp(β),ξnφ:cos_sp(α)/sin_sp(β),ξlA:tan_sp(β),ξnA:cot_sp(β)}
#typeII_ξH = {ξlφ:cos_sp(α)/cos_sp(β),ξnφ:sin_sp(α)/sin_sp(β),ξlA:tan_sp(β),ξnA:cot_sp(β)}

lepton_ξh = {ξlφ:-sin_sp(α)/cos_sp(β),ξnφ:-sin_sp(α)/cos_sp(β),ξlA:tan_sp(β),ξnA:-tan_sp(β)}
#lepton_ξH = {ξlφ:cos_sp(α)/cos_sp(β),ξnφ:cos_sp(α)/cos_sp(β),ξlA:tan_sp(β),ξnA:-tan_sp(β)}

flipped_ξh = {ξlφ:cos_sp(α)/sin_sp(β),ξnφ:-sin_sp(α)/cos_sp(β),ξlA:-cot_sp(β),ξnA:-tan_sp(β)}
#flipped_ξH = {ξlφ:sin_sp(α)/sin_sp(β),ξnφ:cos_sp(α)/cos_sp(β),ξlA:-cot_sp(β),ξnA:-tan_sp(β)}


mA,mH, mh = symbols('m_A,m_H, m_h ',positive=True)
common_factor_h = {mϕ:mh, Ξϕ:sin_sp(β - α), ηϕ:cos_sp(β - α), Kϕ: 4*mA**2 - 3*mh**2- 2*mHpm**2, 
                   Qϕ:mh**2 - 2*mHpm**2 , ρϕ: cos_sp(α + β), Δϕ:cos_sp(α - 3*β)}

#common_factor_H = {mϕ:mH, Ξϕ:cos_sp(β - α), ηϕ:-sin_sp(β - α), Kϕ: 4*mA**2 - 3*mH**2- 2*mHpm**2, 
#                   Qϕ:mH**2 - 2*mHpm**2 , ρϕ: sin_sp(α + β), Δϕ:sin_sp(α - 3*β)}

### Numeric translation 
from OneLoopLFVHD.data import ml

ma,mb = symbols('m_a,m_b',positive=True)
valores_h ={mW:mpf('80.379'),mh:mpf('125.10'),g:(2*mpf('80.379'))/mpf('246')}

cambios_hab = lambda a,b:{lfvhd.ma:valores_h[mh],lfvhd.mi:ml[a],lfvhd.mj:ml[b]}


Ubi, Ucai,mni = symbols('U_{bi}, {{U_{ai}^*}},m_{n_i}')
UnuOne = {mn[i]:mni,Uν[b,i]:Ubi,Uνc[a,i]:Ucai}

# from Unu_seesaw_2HDM import diagonalizationMnu
# diagonalizationMnu1 = lambda m1,m6,tb,doblet: diagonalizationMnu(
#     m1,m2,m3,m6/mpf('3.0'),m6/mpf('2.0'),m6,tb,doblet)

from Unu_seesaw import diagonalizationMnu

diagonalizationMnu1 = lambda m1,m6: diagonalizationMnu(
    m1,m2,m3,m6/mpf('3.0'),m6/mpf('2.0'),m6)

## Internal funtions OneFermion
def GIM_One(exp):
    from sympy import Add
    args = exp.expand().args
    func = exp.expand().func
    if isinstance(func,Add):
        X = Add(*[t for t in args if t.has(mni)]).simplify()
    else:
        X = exp
    #X1 = X.collect([mni],evaluate=False)
    return X#mni**2*X1[mni**2]

def sumOne(m6,Aab,aa,bb,mHpm_val, mA_val, alpha, beta, l5): 
    mnk,UnuL,UnuR = diagonalizationMnu1(m1,m6)
    Unu = UnuL
    Unu_dagger = UnuR
    AL = []
    for k in range(1,7):
        #print(mnk[k-1],Unu[b-1,k-1],conj(Unu[a-1,k-1]))
        #A = Aab(mnk[k-1],Unu[b-1,k-1],conj(Unu[a-1,k-1]),mHpm, mA, alpha, beta,l5)
        A = Aab(mnk[k-1],Unu[bb-1,k-1],Unu_dagger[k-1,aa-1],mHpm_val, mA_val, alpha, beta,l5)
        #print('Ai = ',A)
        AL.append(A)
    return mp.fsum(AL)

from OneLoopLFVHD.data import replaceBs, pave_functions
mHpm_aux = symbols('mHpm',positive=True)
def numeric_sum_diagramsOne(a,b,mHpm_n, mA_n, alpha, beta, l5,quirality='L',
                            common_factor=common_factor_h,
                            type_2HDM=typeI_ξh,
                            valores=valores_h):
    
    FdiagOneFer = []
    i = 0
    #print('Inside numeric_sum_diagramsOne')
    for Set in [TrianglesOneFermion,Bubbles]:#TrianglesOneFermion,Bubbles
        for dia in Set:
            if quirality=='L':
                x = dia.AL().subs(common_factor).subs(type_2HDM).subs(lfvhd.D,4).subs(
                    lfvhd.B12_0(mW,mW),0).subs(lfvhd.B12_0(mHpm,mW),0).subs(
                    lfvhd.B12_0(mW,mHpm),0).subs(cambios_hab(a,b)).subs(valores).subs(UnuOne)
            elif quirality=='R':
                x = dia.AR().subs(common_factor).subs(type_2HDM).subs(lfvhd.D,4).subs(
                    lfvhd.B12_0(mW,mW),0).subs(lfvhd.B12_0(mHpm,mW),0).subs(
                    lfvhd.B12_0(mW,mHpm),0).subs(cambios_hab(a,b)).subs(valores).subs(UnuOne)
            else:
                raise ValueError('quirality must be L or R')

            f = lambdify([mni,Ubi,Ucai, mHpm_aux, mA, α, β,λ5],replaceBs(x).subs(mHpm,mHpm_aux),
                         modules=[pave_functions(valores_h[mh],a,b,lib='mpmath'),'mpmath'])
            #print(f'diagram i = {i}')
            #nprint(f(mpf('100'),0.1,0.2,mHpm_n, mA_n, alpha, beta, l5))
            #fsum = lambda m6:sumOne(m6,f,a,b)
            FdiagOneFer.append(f)
            i+=1
    def suma(m6):
        out = []
        xs = []
        #print('suma sobre i')
        for FF in FdiagOneFer:
            x = sumOne(m6,FF,a,b,mHpm_n, mA_n, alpha, beta, l5)
            #print(x)
            #m6,Aab,a,b,mHpm, mA, alpha, beta, l5,doblet
            out.append(x)
            xs.append(x)
        #print('suma sobre i terminada')
        return np.array(xs), mp.fsum(out)
    return suma

#### Here we can chande the comon factor for h or H and valores h or H
def ALOneTot23(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return numeric_sum_diagramsOne(2,3,mHpm_n, mA_n, alpha,beta, l5,quirality='L',
                            common_factor=common_factor_h,
                            type_2HDM=modelo,
                            valores=valores_h)(m6)[1]

def AROneTot23(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return numeric_sum_diagramsOne(2,3,mHpm_n, mA_n, alpha,beta, l5,quirality='R',
                            common_factor=common_factor_h,
                            type_2HDM=modelo,
                            valores=valores_h)(m6)[1]

def ALOneTot32(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return numeric_sum_diagramsOne(3,2,mHpm_n, mA_n, alpha,beta, l5,quirality='L',
                            common_factor=common_factor_h,
                            type_2HDM=modelo,
                            valores=valores_h)(m6)[1]

def AROneTot32(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return numeric_sum_diagramsOne(3,2,mHpm_n, mA_n, alpha,beta, l5,quirality='R',
                            common_factor=common_factor_h,
                            type_2HDM=modelo,
                            valores=valores_h)(m6)[1]


def ALOneTot13(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return numeric_sum_diagramsOne(1,3,mHpm_n, mA_n, alpha,beta, l5,quirality='L',
                            common_factor=common_factor_h,
                            type_2HDM=modelo,
                            valores=valores_h)(m6)[1]

def AROneTot13(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return numeric_sum_diagramsOne(1,3,mHpm_n, mA_n, alpha,beta, l5,quirality='R',
                            common_factor=common_factor_h,
                            type_2HDM=modelo,
                            valores=valores_h)(m6)[1]

def ALOneTot31(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return numeric_sum_diagramsOne(3,1,mHpm_n, mA_n, alpha,beta, l5,quirality='L',
                            common_factor=common_factor_h,
                            type_2HDM=modelo,
                            valores=valores_h)(m6)[1]

def AROneTot31(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return numeric_sum_diagramsOne(3,1,mHpm_n, mA_n, alpha,beta, l5,quirality='R',
                            common_factor=common_factor_h,
                            type_2HDM=modelo,
                            valores=valores_h)(m6)[1]


def ALOneTot12(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return numeric_sum_diagramsOne(1,2,mHpm_n, mA_n, alpha,beta, l5,quirality='L',
                            common_factor=common_factor_h,
                            type_2HDM=modelo,
                            valores=valores_h)(m6)[1]

def AROneTot12(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return numeric_sum_diagramsOne(1,2,mHpm_n, mA_n, alpha,beta, l5,quirality='R',
                            common_factor=common_factor_h,
                            type_2HDM=modelo,
                            valores=valores_h)(m6)[1]

def ALOneTot21(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return numeric_sum_diagramsOne(2,1,mHpm_n, mA_n, alpha,beta, l5,quirality='L',
                            common_factor=common_factor_h,
                            type_2HDM=modelo,
                            valores=valores_h)(m6)[1]

def AROneTot21(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return numeric_sum_diagramsOne(2,1,mHpm_n, mA_n, alpha,beta, l5,quirality='R',
                            common_factor=common_factor_h,
                            type_2HDM=modelo,
                            valores=valores_h)(m6)[1]

## Internal funtions TwoFermion

mnj = symbols('m_{n_j}',positive=True)
Cijs, Cijcs, Ubj = symbols('C_{ij}, {{C_{ij}^*}},U_{bj}')
UnuTwo = {mn[i]:mni,mn[jj]:mnj,C[i,jj]:Cijs, Cc[i,jj]:Cijcs, Uν[b,jj]:Ubj, Uνc[a,i]:Ucai}


def FFsymbolic(k,a,b,quirality='L',common_factor=common_factor_h,
               type_2HDM=typeI_ξh,valores=valores_h):
    if quirality=='L':
        FF = TrianglesTwoFermion[k].AL()
    elif quirality=='R':
        FF = TrianglesTwoFermion[k].AR()
    else:
        raise ValueError('quirality must be L or R')
    return FF.subs(common_factor).subs(type_2HDM).subs(lfvhd.D,4).subs(
        cambios_hab(a,b)).subs(valores).subs(UnuTwo)

FFmpL = lambda k,a,b,modelo:lambdify([mni,mnj,Ubj,Ucai,Cijs,Cijcs,mHpm_aux,α,β],
                              replaceBs(FFsymbolic(k,a,b,quirality='L'
                            ,common_factor=common_factor_h,
                            type_2HDM=modelo#########
                            ,valores=valores_h).subs(mHpm,mHpm_aux)),
                     modules=[pave_functions(valores_h[mh],a,b,lib='mpmath'),'mpmath'] )

FFmpR = lambda k,a,b,modelo:lambdify([mni,mnj,Ubj,Ucai,Cijs,Cijcs,mHpm_aux,α,β],
                              replaceBs(FFsymbolic(k,a,b,quirality='R'
                            ,common_factor=common_factor_h,
                            type_2HDM=modelo#########
                            ,valores=valores_h).subs(mHpm,mHpm_aux)),
                     modules=[pave_functions(valores_h[mh],a,b,lib='mpmath'),'mpmath'] )

def sumatwo(mm6,k,a,b,mHpm_n,alpha,beta,modelo,quirality='L'):
    xs = []
    if quirality=='L':
        g = FFmpL(k,a,b,modelo)
    elif quirality=='R':
        g = FFmpR(k,a,b,modelo)
    else:
        raise ValueError('quirality must be L or R')
        
    mnk,UnuL, UnuR = diagonalizationMnu1(m1,mm6)
    Unu = UnuL
    Unu_dagger = UnuR
    #Cij = lambda i,j: mp.fsum([Unu[c,i]*conj(Unu[c,j]) for c in range(3)])
    Cij = lambda i,j: mp.fsum([Unu[c,i]*Unu_dagger[j,c] for c in range(3)])
    for p in range(1,7):
        for q in range(1,7):
            #x = g(mnk[p-1],mnk[q-1],Unu[b-1,q-1],conj(Unu[a-1,p-1]),Cij(p-1,q-1),conj(Cij(p-1,q-1)),mHpm_n,alpha,beta)
            x = g(mnk[p-1],mnk[q-1],Unu[b-1,q-1],Unu_dagger[p-1,a-1],Cij(p-1,q-1),conj(Cij(p-1,q-1)),mHpm_n,alpha,beta)
            xs.append(x)
            #print(f'i = {p} and j = {q}')
            #print(f'|f| = {x}')
    return mp.fsum(xs)

def totaltwo(m6,a,b,mHpm_n,alpha,beta,modelo,quirality='L'):
    #print('Inside totaltwo')
    #print('doble suma sobre i j')
    out = (sumatwo(m6,0,a,b,mHpm_n,alpha,beta,modelo,quirality) + sumatwo(m6,1,a,b,mHpm_n,alpha,beta,modelo,quirality)
           + sumatwo(m6,2,a,b,mHpm_n,alpha,beta,modelo,quirality))
    #print('doble suma sobre i j terminada')
    return out

def ALTwoTot23(m6,mHpm_n,alpha,beta,modelo):
    return totaltwo(m6,2,3,mHpm_n,alpha,beta,modelo,quirality='L')
def ARTwoTot23(m6,mHpm_n,alpha,beta,modelo): 
    return totaltwo(m6,2,3,mHpm_n,alpha,beta,modelo,quirality='R')

def ALTwoTot32(m6,mHpm_n,alpha,beta,modelo): 
    return totaltwo(m6,3,2,mHpm_n,alpha,beta,modelo,quirality='L')
def ARTwoTot32(m6,mHpm_n,alpha,beta,modelo): 
    return totaltwo(m6,3,2,mHpm_n,alpha,beta,modelo,quirality='R')

def ALTwoTot13(m6,mHpm_n,alpha,beta,modelo): 
    return totaltwo(m6,1,3,mHpm_n,alpha,beta,modelo,quirality='L')
def ARTwoTot13(m6,mHpm_n,alpha,beta,modelo): 
    return totaltwo(m6,1,3,mHpm_n,alpha,beta,modelo,quirality='R')

def ALTwoTot31(m6,mHpm_n,alpha,beta,modelo): 
    return totaltwo(m6,3,1,mHpm_n,alpha,beta,modelo,quirality='L')
def ARTwoTot31(m6,mHpm_n,alpha,beta,modelo): 
    return totaltwo(m6,3,1,mHpm_n,alpha,beta,modelo,quirality='R')
def ALTwoTot12(m6,mHpm_n,alpha,beta,modelo): 
    return totaltwo(m6,1,2,mHpm_n,alpha,beta,modelo,quirality='L')
def ARTwoTot12(m6,mHpm_n,alpha,beta,modelo): 
    return totaltwo(m6,1,2,mHpm_n,alpha,beta,modelo,quirality='R')

def ALTwoTot21(m6,mHpm_n,alpha,beta,modelo): 
    return totaltwo(m6,2,1,mHpm_n,alpha,beta,modelo,quirality='L')
def ARTwoTot21(m6,mHpm_n,alpha,beta,modelo): 
    return totaltwo(m6,2,1,mHpm_n,alpha,beta,modelo,quirality='R')

################3
# Total Form Factor
###################
#a = 2, b = 3
def ALtot23(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return  ALOneTot23(m6,mHpm_n, mA_n, alpha, beta, l5,modelo) + ALTwoTot23(m6,mHpm_n,alpha,beta,modelo)
def ARtot23(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return  AROneTot23(m6,mHpm_n, mA_n, alpha, beta, l5,modelo) + ARTwoTot23(m6,mHpm_n,alpha,beta,modelo)

#a = 3, b = 2
def ALtot32(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return  ALOneTot32(m6,mHpm_n, mA_n, alpha, beta, l5,modelo) + ALTwoTot32(m6,mHpm_n,alpha,beta,modelo)
def ARtot32(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return  AROneTot32(m6,mHpm_n, mA_n, alpha, beta, l5,modelo) + ARTwoTot32(m6,mHpm_n,alpha,beta,modelo)

#a = 1, b = 3
def ALtot13(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return  ALOneTot13(m6,mHpm_n, mA_n, alpha, beta, l5,modelo) + ALTwoTot13(m6,mHpm_n,alpha,beta,modelo)
def ARtot13(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return  AROneTot13(m6,mHpm_n, mA_n, alpha, beta, l5,modelo) + ARTwoTot13(m6,mHpm_n,alpha,beta,modelo)

#a = 3, b = 1
def ALtot31(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return  ALOneTot31(m6,mHpm_n, mA_n, alpha, beta, l5,modelo) + ALTwoTot31(m6,mHpm_n,alpha,beta,modelo)
def ARtot31(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return  AROneTot31(m6,mHpm_n, mA_n, alpha, beta, l5,modelo) + ARTwoTot31(m6,mHpm_n,alpha,beta,modelo)

#a = 1, b = 2
def ALtot12(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return  ALOneTot12(m6,mHpm_n, mA_n, alpha, beta, l5,modelo) + ALTwoTot12(m6,mHpm_n,alpha,beta,modelo)
def ARtot12(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return  AROneTot12(m6,mHpm_n, mA_n, alpha, beta, l5,modelo) + ARTwoTot12(m6,mHpm_n,alpha,beta,modelo)

#a = 2, b = 1
def ALtot21(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return  ALOneTot21(m6,mHpm_n, mA_n, alpha, beta, l5,modelo) + ALTwoTot21(m6,mHpm_n,alpha,beta,modelo)
def ARtot21(m6,mHpm_n, mA_n, alpha, beta, l5,modelo):
    return  AROneTot21(m6,mHpm_n, mA_n, alpha, beta, l5,modelo) + ARTwoTot21(m6,mHpm_n,alpha,beta,modelo)


### DEf angles alpha and beta
def betaf(tb):
    return mp.atan(tb)
def alphaf(tb,x0=mp.mpf('0.01')): #Alignment h SM-like.
    return mp.atan(tb) - mp.acos(x0)
###############################################
# we set cos(beta - alpha)=0.01 and create the cases
###############################################
caso1 = {'mA':'800','mHpm':'1000.0','l5':'0.1', 'mn6':'1e10','n':'1'}
caso2 = {'mA':'800','mHpm':'1000.0','l5':'1', 'mn6':'1e10','n':'2'}
caso3 = {'mA':'1300','mHpm':'1500.0','l5':'1', 'mn6':'1e10','n':'3'}
caso4 = {'mA':'1300','mHpm':'1500.0','l5':'1', 'mn6':'1e15','n':'4'}

################################################
### We set com setings of each model to automatize
################################################
modelo_typeI = lambda caso: {'xi_factors':typeI_ξh,'carpeta':'Type_I',
                             'name':f"TypeI_Cab095_caso{caso['n']}"}
modelo_typeII = lambda caso: {'xi_factors':typeII_ξh,'carpeta':'Type_II',
                             'name':f"TypeII_Cab095_caso{caso['n']}"}
modelo_lepton = lambda caso: {'xi_factors':lepton_ξh,'carpeta':'Lepton_specific',
                              'name':f"Lepton_specific_Cab095_caso{caso['n']}"}
modelo_flipped = lambda caso: {'xi_factors':flipped_ξh,'carpeta':'Flipped',
                              'name':f"Flipped_Cab095_caso{caso['n']}"}

modelos = [modelo_typeI,modelo_typeII,modelo_lepton,modelo_flipped]
casos = [caso1, caso2, caso3, caso4]

if __name__ == '__main__':
    #nprint((abs(ALTwoTot12(0.1,1,2,3,typeI_ξh))))
    #nprint(ALtot12(m1,1,2,3,4,5,typeI_ξh)),label=r'Br($h \to e \tau$)'

    
    def ALtot23_caso1(tb,m6,mHpm_n,mA_n,alpha,beta, l5,modelo):
        return ALtot23(m6,mHpm_n,mA_n,alpha,beta, l5,modelo)
    def ARtot23_caso1(tb,m6,mHpm_n,mA_n,alpha,beta, l5,modelo):
        return ARtot23(m6,mHpm_n,mA_n,alpha,beta, l5,modelo)
    def ALtot32_caso1(tb,m6,mHpm_n,mA_n,alpha,beta, l5,modelo):
        return ALtot32(m6,mHpm_n,mA_n,alpha,beta, l5,modelo)
    def ARtot32_caso1(tb,m6,mHpm_n,mA_n,alpha,beta, l5,modelo):
        return ARtot32(m6,mHpm_n,mA_n,alpha,beta, l5,modelo)
    def ALtot13_caso1(tb,m6,mHpm_n,mA_n,alpha,beta, l5,modelo):
        return ALtot13(m6,mHpm_n,mA_n,alpha,beta, l5,modelo)
    def ARtot13_caso1(tb,m6,mHpm_n,mA_n,alpha,beta, l5,modelo):
        return ARtot13(m6,mHpm_n,mA_n,alpha,beta, l5,modelo)
    def ALtot31_caso1(tb,m6,mHpm_n,mA_n,alpha,beta, l5,modelo):
        return ALtot31(m6,mHpm_n,mA_n,alpha,beta, l5,modelo)
    def ARtot31_caso1(tb,m6,mHpm_n,mA_n,alpha,beta, l5,modelo):
        return ARtot31(m6,mHpm_n,mA_n,alpha,beta, l5,modelo)
    def ALtot12_caso1(tb,m6,mHpm_n,mA_n,alpha,beta, l5,modelo):
        return ALtot12(m6,mHpm_n,mA_n,alpha,beta, l5,modelo)
    def ARtot12_caso1(tb,m6,mHpm_n,mA_n,alpha,beta, l5,modelo):
        return ARtot12(m6,mHpm_n,mA_n,alpha,beta, l5,modelo)
    def ALtot21_caso1(tb,m6,mHpm_n,mA_n,alpha,beta, l5,modelo):
        return ALtot21(m6,mHpm_n,mA_n,alpha,beta, l5,modelo)
    def ARtot21_caso1(tb,m6,mHpm_n,mA_n,alpha,beta, l5,modelo):
        return ARtot21(m6,mHpm_n,mA_n,alpha,beta, l5,modelo)
    
    ### Width decay
    from OneLoopLFVHD import Γhlilj 
    from pandas import DataFrame
    import matplotlib.pyplot as plt
    n = 50#800
    expmp = linspace(-2,2,n)
    tbmp = np.array([mpf('10.0')**k for k in expmp])#np.logspace(-1,15,n)
    #y1 = np.ones_like(tbmp)
    Atlas_bound13 = 0.00047
    Atlas_bound23 = 0.00028
    #y2 = np.ones_like(tbmp)*Atlas_bound23
    
    for modelo in modelos:
        for caso in casos:

            mHpm_val = mp.mpf(caso['mHpm'])
            mA_val = mp.mpf(caso['mA']) 
            l5_val = mp.mpf(caso['l5']) 
            m6_val = mp.mpf(caso['mn6'])
            ######################################
            ###### here we choose the model
            ######################################
            model = modelo(caso)#modelo_lepton(caso) 

            def Γhl2l3(tb):
                return Γhlilj(ALtot23_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                            modelo=model['xi_factors']),
                              ARtot23_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                            modelo=model['xi_factors']),
                              valores_h[mh],ml[2],ml[3])
            def Γhl3l2(tb):
                return Γhlilj(ALtot32_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                            modelo=model['xi_factors']),
                              ARtot32_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                            modelo=model['xi_factors']),
                              valores_h[mh],ml[3],ml[2])
            def Γhl1l3(tb):
                return Γhlilj(ALtot13_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                            modelo=model['xi_factors']),
                              ARtot13_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                            modelo=model['xi_factors']),
                              valores_h[mh],ml[1],ml[3])
            def Γhl3l1(tb):
                return Γhlilj(ALtot31_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                            modelo=model['xi_factors']),
                              ARtot31_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                            modelo=model['xi_factors']),
                              valores_h[mh],ml[3],ml[1])
            def Γhl1l2(tb):
                return Γhlilj(ALtot12_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                            modelo=model['xi_factors']),
                              ARtot12_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                            modelo=model['xi_factors']),
                              valores_h[mh],ml[1],ml[2])
            def Γhl2l1(tb):
                return Γhlilj(ALtot21_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                            modelo=model['xi_factors']),
                              ARtot21_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                            modelo=model['xi_factors']),
                              valores_h[mh],ml[2],ml[1])


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

            path = f"output/{model['carpeta']}/"
            plt.savefig(path+f"{model['name']}.png",dpi=100)
            plt.show()

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

            df.to_csv(path+f"{model['name']}.txt",sep='\t')
            print(path+f"{model['name']}.txt has been completed")