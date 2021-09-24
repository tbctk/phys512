import numpy as np

def log_2(x,errtol=1e-6):
    #assert(x > 0) I commented this out to make the code work for arrays
    n = 30
    # For chebyshev we need to rescale the x-axis to between -1 and 1
    # For x from 0.5 to 1, u = 4*x - 3 works. Then we can use the built in
    # numpy functions to get the approximation.
    xx = np.linspace(0.5,1,1001)
    yy = np.log2(xx)
    uu = 4*xx-3
    coeffs = np.polynomial.chebyshev.chebfit(uu,yy,n)
    # This bit of code truncates as many terms as possible in the coefficient
    # array such that the max error remains less than errtol (1e-6 here).
    i = n-1
    err = coeffs[i]
    while i > 0 and err < errtol:
        i = i-1
        err = err + coeffs[i]
    coeffs = coeffs[0:i+1]
    # print(len(coeffs))
    u = 4*x-3
    return np.polynomial.chebyshev.chebval(u,coeffs)

def mylog2(x):
    #assert(x > 0)
    ln2 = np.log(2)
    man,exp = np.frexp(x)
    # In floating point, a number is represented as mantissa (man) times
    # 2^exponent (exp). So for x = man*2^exp, we can write
    # ln(x) = ln(man) + exp*ln(2). Then we can write ln(man) = ln(2)*log_2(man).
    # Since man is guaranteed to be between 0.1 and 1 in binary (0.5 and 1 in
    # base 10), we can use our log_2 function from before to solve using the
    # Chebyshev approximation. Then all that is left is the constant ln(2) which
    # must only be evaluated once.
    return ln2*(exp + log_2(man))
