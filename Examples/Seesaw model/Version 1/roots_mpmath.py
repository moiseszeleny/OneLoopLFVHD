from mpmath import *
mp.dps = 35; mp.pretty = True

ma = 125.1; M1 = M2 = 1e-10;
a = mpf(1.0)
b = - ((ma**2 -M1**2 + M2**2)/ma**2)
c = (M2**2)/ma**2

r = b**2 - 4*a*c
x1 = (-b - sqrt(r))/(2*a)
x2 = (-b + sqrt(r))/(2*a)

f = lambda x: a*x**2 + b*x + c

x = findroot(f,1.1)
print(type(x))
print('x = ',x)
print('x1 = ',x1)
print('x2 = ',x2)
print('polylog(2,x) = ',polylog(2,x))
print('log(x) = ',log(x))
print('1-1/x = ',1-1/x)
print('log(1-1/x) = ',log(1-1/x))
