import numpy as np

x = 42
eps = 2**-52

def deriv(fun,x,dx):
    dx = (x+dx)-x
    f1 = fun(x+dx)
    f2 = fun(x-dx)
    f3 = fun(x+2*dx)
    f4 = fun(x-2*dx)
    return (8*f1-8*f2-f3+f4)/(12*dx)

opt = eps**(1/5)

print('For f(x) = exp(x), x = 42:')
print('Relative error with optimal dx: ',(deriv(np.exp,x,opt)-np.exp(x))/np.exp(x))
print('Relative error with 10 times larger dx: ',(deriv(np.exp,x,opt*10)-np.exp(x))/np.exp(x))
print('Relative error with 10 times smaller dx: ',(deriv(np.exp,x,opt/10)-np.exp(x))/np.exp(x))

def fun(x):
    return np.exp(0.01*x)

print('\nFor f(x) = exp(0.01x), x = 42:')
print('Relative error with optimal dx: ',(deriv(fun,x,opt)-0.01*fun(x))/(0.01*fun(x)))
print('Relative error with 10 times larger dx: ',(deriv(fun,x,opt*10)-0.01*fun(x))/(0.01*fun(x)))
print('Relative error with 10 times smaller dx: ',(deriv(fun,x,opt/10)-0.01*fun(x))/(0.01*fun(x)))

