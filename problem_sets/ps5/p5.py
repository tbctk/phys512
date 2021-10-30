import numpy as np

## PROBLEM 5

def anal_sin_dft(k_0,N,k):
    top1 = 1-np.exp(-2j*np.pi*(k-k_0))
    bot1 = 1-np.exp(-2j*np.pi*(k-k_0)/N)
    top2 = 1-np.exp(-2j*np.pi*(k+k_0))
    bot2 = 1-np.exp(-2j*np.pi*(k+k_0)/N)
    return (top1/bot1 - top2/bot2)/2j

def win(x):
    return 0.5-0.5*np.cos(2*np.pi*x/len(N))
