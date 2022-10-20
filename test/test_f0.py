# from nose.tools import assert_equal

from mpmath import mpf, mpc
from OneLoopLFVHD.LFVHDFeynG_mpmathDelta import f0np, f1np
from OneLoopLFVHD.roots_mpmath2 import y11, y12  # , y21, y22
from OneLoopLFVHD import y11 as y11sp
from OneLoopLFVHD import y12 as y12sp
from OneLoopLFVHD import y21 as y21sp
from OneLoopLFVHD import y22 as y22sp
from OneLoopLFVHD import f0 as f0sp
from OneLoopLFVHD import f1 as f1sp
from OneLoopLFVHD import mi, ma, M0, M1, M2, y
from OneLoopLFVHD.data import ml
from sympy import S


def test_f0_0():
    '''test f0(1)'''
    obs = f0np(mpf('1.0'))
    exp = mpc(complex(f0sp(1).n()))
    # assert_equal(exp, obs)
    assert exp == obs

def test_f1_1():
    '''test f1(1.0)'''
    obs = f1np(mpf('1.0'))
    exp = mpc(complex(f1sp(1).n()))
    # assert_equal(exp, obs)
    assert exp == obs

def test_y11_0():
    '''test y11(mi, M0, 0)'''
    mW = mpf('80.379')
    obs = y11(ml[2], mW, 0)
    exp = y11sp.evalf(18,{mi: ml[2], M0: mW, M1:0}).n()
    # assert_equal(exp, obs)
    assert abs(exp - obs)<1e-8

def test_y11_1():
    '''test y11(mi, 0, M1)'''
    mW = mpf('80.379')
    obs = y11(ml[2], 0, mW)
    exp = y11sp.evalf(18,{mi: ml[2], M0: 0, M1: mW}).n()
    # assert_equal(exp, obs)
    assert abs(exp - obs)<1e-8

def test_y11_2():
    '''test y11(mi, M0, M1)'''
    mW = mpf('80.379')
    MN = mpf('1e15')
    obs = y11(ml[2], mW, MN)
    exp = y11sp.evalf(18,{mi: ml[2], M0: mW, M1:MN}).n()
    # assert_equal(exp, obs)
    assert abs(exp - obs)<1e-8

def test_y12_0():
    '''test y12(mi, M0, 0)'''
    mW = mpf('80.379')
    obs = y12(ml[2], mW, 0)
    exp = y12sp.evalf(32,{mi: ml[2], M0: mW, M1:0}).n()
    # assert_equal(exp, obs)
    assert abs(exp - obs)<1e-8

def test_y12_1():
    '''test y12(mi, 0, M1)'''
    mW = mpf('80.379')
    obs = y12(ml[2], 0, mW)
    exp = y12sp.evalf(32,{mi: ml[2], M0: 0, M1:mW}).n()
    # assert_equal(exp, obs)
    assert abs(exp - obs)<1e-8

def test_y12_2():
    '''test y11(mi, M0, M1)'''
    mW = mpf('80.379')
    MN = mpf('1e15')
    obs = y12(ml[2], mW, MN)
    exp = y12sp.evalf(18,{mi: ml[2], M0: mW, M1:MN}).n()
    # assert_equal(exp, obs)
    assert abs(exp - obs)<1e-8

def test_f0_y11_0():
    '''test f0(y11(mi,M0,0))'''
    mW = mpf('80.379')
    obs = f0np(y11(ml[2], mW, 0))
    # y11val = y11sp.evalf(18,{mi: ml[2], M0: mW, M1:0})
    exp = f0sp(y11sp).evalf(18,{mi: ml[2], M0: mW, M1:0})
    # assert_equal(exp, obs)
    assert abs(exp - obs) < 1e-8

def test_f0_y11_1():
    '''test f0(y11(mi,M0,0))'''
    mW = mpf('80.379')
    obs = f0np(y11(ml[2], 0, mW))
    # Y11val = y11sp.evalf(18,{mi: ml[2], M0: 0, M1:mW})
    exp = f0sp(y11sp).evalf(18,{mi: ml[2], M0: 0, M1:mW})
    # assert_equal(exp, obs)
    assert abs(exp - obs) < 1e-8