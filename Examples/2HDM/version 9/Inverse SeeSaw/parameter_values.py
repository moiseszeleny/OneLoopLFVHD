from mpmath import mpf

m6 = mpf('1e3')
mux0 = mpf('1e-7')
mh = mpf('125.1')

Benchmarck1 = {'mA':'300','mHpm':'500.0','l5':'1', 'cab':'0.01', 'n':'1'}
Benchmarck2 = {'mA':'300','mHpm':'500.0','l5':'1', 'cab':'0.001', 'n':'2'}
Benchmarck3 = {'mA':'200','mHpm':'300.0','l5':'1', 'cab':'0.01', 'n':'3'}
Benchmarck4 = {'mA':'200','mHpm':'300.0','l5':'1', 'cab':'0.001', 'n':'4'}

casos1 = [Benchmarck1, Benchmarck2] #, Benchmarck3, Benchmarck4]

n_points = 15
# Exponential values of tan(b)
tbi = -3
tbf = 3
