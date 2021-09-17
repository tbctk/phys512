import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

npt = 11
x = np.linspace(-np.pi/2,np.pi/2,npt)
y_true = np.cos(x)
dx = x[1]-x[0]

plt.figure()
plt.plot(x,y_true,'r+')

# Polynomial interpolation
X = np.empty([npt,npt])
for i in range(npt):
    X[:,i]=x**i
Xinv = np.linalg.inv(X)
c = Xinv@y_true
y_poly = X@c

err = np.std(y_true-y_poly)
print('Polynomial interpolation error: ',err)

plt.plot(x,y_poly)

# Spline interpolation
spln = interpolate.splrep(x,y_true)
y_spln = interpolate.splev(x,spln)

err = np.std(y_true-y_spln)
print('Spline interpolation error: ',err)

plt.plot(x,y_spln)

# Rational function interpolation
# The rational function interpolation doesn't seem to be working for me
def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.inv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q

n,m = 6,6
p,q = rat_fit(x,y_true,n,m)
y_rat = rat_eval(p,q,x)

err = np.std(y_true-y_rat)
print('Rational function interpolation error: ',err)

plt.plot(x,y_rat)



plt.show()
