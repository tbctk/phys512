import numpy as np

num_func_calls = 0

def integrate_adaptive(fun,a,b,tol,extra=None):
    global num_func_calls
    x = np.linspace(a,b,5)
    dx = x[1]-x[0]
    if extra == None:
        f0 = fun(a)
        f4 = fun(b)
        f2 = fun(x[2])
        num_func_calls = num_func_calls + 3
    else:
        (f0,f4,f2) = extra
    sol0 = 2*(f0+4*f2+f4)*dx/3
    
    f1,f3 = fun(x[1]),fun(x[3])
    num_func_calls = num_func_calls + 2

    sol1 = (f0+4*f1+2*f2+4*f3+f4)*dx/3
    
    if np.abs(sol1-sol0) < tol:
        return sol1
    else:
        return integrate_adaptive(fun,x[0],x[2],tol/2,extra=(f0,f2,f1))+integrate_adaptive(fun,x[2],x[4],tol/2,extra=(f2,f4,f3))

def fun(x):
    return 1/(1+x**2)

t = integrate_adaptive(fun,-10,10,1e-9)
print('Error on numerical integral: ',t -(np.arctan(10)-np.arctan(-10)))

# By adding the 'extra' parameter, we are able to reduce the number of function calls by 3 with each recursive call,
# Resulting in a total decrease of 6 calls per "level". In our implementation we only do 2 function calls per recursive
# call, or 4 function calls per level. To illustrate this, I added the variable "num_func_calls" to
# keep track of the total number of function calls and will do the same without the extra variable to compare.
# If my prediction is correct, we should see about 2/5 (0.4x) the number of function calls in the first implementation.

num_func_calls2 = 0

def integrate_adaptive2(fun,a,b,tol):
    global num_func_calls2
    
    x = np.linspace(a,b,5)
    dx = x[1]-x[0]
    f0 = fun(a)
    f4 = fun(b)
    f2 = fun(x[2])
    num_func_calls2 = num_func_calls2 + 3
    
    sol0 = 2*(f0+4*f2+f4)*dx/3
    
    f1,f3 = fun(x[1]),fun(x[3])
    num_func_calls2 = num_func_calls2 + 2

    sol1 = (f0+4*f1+2*f2+4*f3+f4)*dx/3
    
    if np.abs(sol1-sol0) < tol:
        return sol1
    else:
        return integrate_adaptive2(fun,x[0],x[2],tol/2)+integrate_adaptive2(fun,x[2],x[4],tol/2)

t2 = integrate_adaptive2(fun,-10,10,1e-9)

print('# function calls with extra variable: ',num_func_calls)
print('# function call without extra variable: ',num_func_calls2)
print('Ratio: ',num_func_calls/num_func_calls2)


