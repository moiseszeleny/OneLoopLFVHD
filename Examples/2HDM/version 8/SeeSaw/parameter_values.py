from mpmath import mpf

m6 = mpf('1e15')
mh = mpf('125.1')

Benchmarck1 = {'mA':'300','mHpm':'500.0','l5':'1', 'cab':'0.01', 'n':'1'}
Benchmarck2 = {'mA':'300','mHpm':'500.0','l5':'1', 'cab':'0.01', 'n':'2'}
Benchmarck3 = {'mA':'1300','mHpm':'1500.0','l5':'1', 'cab':'0.01', 'n':'3'}
Benchmarck4 = {'mA':'1300','mHpm':'1500.0','l5':'1', 'cab':'0.0', 'n':'4'}

# casos = [Benchmarck1, Benchmarck2, Benchmarck3, Benchmarck4]
casos1 = [Benchmarck3, Benchmarck4]

n_points = 15
# Exponential values of tan(b)
tbi = -2
tbf = 2