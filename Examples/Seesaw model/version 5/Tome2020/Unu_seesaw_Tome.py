'''
    DEfinitions involved in neutrino masses in SeeSaw model
'''
from mpmath import mp
import numpy as np
from scipy.linalg import eig as eig_sp
from scipy.linalg import svd as svd_sp
LA = np.linalg

##########################################
####### MPMATH  #############################


# def Ynu(m1, m2, m3, m4, m5, m6):
#     '''
#         Yukawa matrix in the SeeSaw model.
#     '''
#     # v = mp.mpf('246')
#     return MD_NO(m1, m2, m3, m4, m5, m6)*(mp.sqrt('2')/v)


def Mnu(m1, m2, m3, M, mu):
    '''
        Neutrino mass matrix in the SeeSaw model.
    '''
    return mp.matrix([
        [0, 0, 0, 0, m1],
        [0, 0, 0, 0, m2],
        [0, 0, 0, 0, m3],
        [0, 0, 0, 0, M],
        [m1, m2, m3, M, mu]
    ])


def diagonalizationMnu(m1, m2, m3, M, mu):
    '''
        Diagonalization of neutrino mass matrix in the SeeSaw model.
    '''
    Mi, UL, UR = mp.eig(Mnu(m1, m2, m3, M, mu), left=True, right=True)
    Mi, UL, UR = mp.eig_sort(Mi, UL, UR)
    return Mi, UL, UR


def diagonalizationMnu_svd(m1, m2, m3, M, mu):
    '''
        Diagonalization of neutrino mass matrix in the SeeSaw model using the
        singular value descomposition method.
    '''
    UL, Mi, UR = mp.svd_c(Mnu(m1, m2, m3, M, mu))
    return Mi, UL, UR

# print(diagonalizationMnu_np(1,2,3,4,5,6))


# if __name__ == '__main__':
#     print(1)