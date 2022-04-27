from FF_numeric_2HDM import ALtot23, ARtot23, ALtot13, ARtot13, ALtot12, ARtot12
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

# from OneLoopLFVHD.neutrinos import NuOscObservables
# Nudata = NuOscObservables

# m1 = mpf('1e-12')  #GeV 

# #current values to Square mass differences
# d21 = mpf(str(Nudata.squareDm21.central))*mpf('1e-18')# factor to convert eV^2 to GeV^2
# d31 = mpf(str(Nudata.squareDm31.central))*mpf('1e-18')

# #d21 = 7.5e-5*1e-18
# #d31 = 2.457e-3*1e-18
# m2 = sqrt(m1**2 + d21)
# m3 = sqrt(m1**2 + d31)
# #######
# m4 = lambda m6: m6/3
# m5 = lambda m6: m6/2


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
modelo_typeI = lambda caso: {'type':'I','carpeta':'Type_I',
                             'filename':f"TypeI_Cab095_caso{caso['n']}"}
modelo_typeII = lambda caso: {'type':'II','carpeta':'Type_II',
                             'filename':f"TypeII_Cab095_caso{caso['n']}"}
modelo_lepton = lambda caso: {'type':'Lepton-specific','carpeta':'Lepton_specific',
                              'filename':f"Lepton_specific_Cab095_caso{caso['n']}"}
modelo_flipped = lambda caso: {'type':'Flipped','carpeta':'Flipped',
                              'filename':f"Flipped_Cab095_caso{caso['n']}"}

# modelos = ['I','II','Lepton-specific','Flipped']
modelos = [modelo_typeI,modelo_typeII,modelo_lepton ,modelo_flipped]

casos = [Benchmarck1, Benchmarck2, Benchmarck3, Benchmarck4]

def ALtot23_caso1(tb,m6,mHpm,mA,alpha,beta, l5,H_a='h',type_2HDM='I'):
        return ALtot23(mh,m6,mHpm, mA, alphaf(tb), betaf(tb), l5,H_a=H_a,type_2HDM=type_2HDM)
    def ARtot23_caso1(tb,m6,mHpm,mA,alpha,beta, l5,H_a='h',type_2HDM='I'):
        return ARtot23(mh,m6,mHpm, mA, alphaf(tb), betaf(tb), l5,H_a=H_a,type_2HDM=type_2HDM)
    
    def ALtot13_caso1(tb,m6,mHpm,mA,alpha,beta, l5,H_a='h',type_2HDM='I'):
        return ALtot13(mh,m6,mHpm, mA, alphaf(tb), betaf(tb), l5,H_a=H_a,type_2HDM=type_2HDM)
    def ARtot13_caso1(tb,m6,mHpm,mA,alpha,beta, l5,H_a='h',type_2HDM='I'):
        return ARtot13(mh,m6,mHpm, mA, alphaf(tb), betaf(tb), l5,H_a=H_a,type_2HDM=type_2HDM)
    
    def ALtot12_caso1(tb,m6,mHpm,mA,alpha,beta, l5,H_a='h',type_2HDM='I'):
        return ALtot12(mh,m6,mHpm, mA, alphaf(tb), betaf(tb), l5,H_a=H_a,type_2HDM=type_2HDM)
    def ARtot12_caso1(tb,m6,mHpm,mA,alpha,beta, l5,H_a='h',type_2HDM='I'):
        return ARtot12(mh,m6,mHpm, mA, alphaf(tb), betaf(tb), l5,H_a=H_a,type_2HDM=type_2HDM)

if __name__ == '__main__':
    #nprint((abs(ALTwoTot12(0.1,1,2,3,typeI_ξh))))
    #nprint(ALtot12(m1,1,2,3,4,5,typeI_ξh)),label=r'Br($h \to e \tau$)'

    mh = mpf('125.1')# SM-like Higgs mass
    
    
    ### Width decay
    from OneLoopLFVHD import Γhlilj 
    from pandas import DataFrame
    import matplotlib.pyplot as plt
    n = 3#800
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

            def Γhl2l3(tb):
                return Γhlilj(ALtot23_caso1(tb,m6=m6_val,mHpm=mHpm_val,mA=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                           H_a='h',type_2HDM=modelo['type']),
                              ARtot23_caso1(tb,m6=m6_val,mHpm=mHpm_val,mA=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                           H_a='h',type_2HDM=modelo['type']),
                              mh,ml[2],ml[3])
            def Γhl1l3(tb):
                return Γhlilj(ALtot13_caso1(tb,m6=m6_val,mHpm=mHpm_val,mA=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                           H_a='h',type_2HDM=modelo['type']),
                              ARtot13_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                           H_a='h',type_2HDM=modelo['type']),
                              mh,ml[1],ml[3])
            def Γhl1l2(tb):
                return Γhlilj(ALtot12_caso1(tb,m6=m6_val,mHpm=mHpm_val,mA=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                           H_a='h',type_2HDM=modelo['type']),
                              ARtot12_caso1(tb,m6=m6_val,mHpm_n=mHpm_val,mA_n=mA_val,
                                            alpha=alphaf(tb),beta=betaf(tb), l5=l5_val,
                                           H_a='h',type_2HDM=modelo['type']),
                              mh,ml[1],ml[2])
            


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

            df.to_csv(path+f"{model['name']}.txt",sep='\t')
            print(path+f"{model['name']}.txt has been completed")
    plt.show()
    