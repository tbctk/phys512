import numpy as np
from matplotlib import pyplot as plt

def win(x):
    return 0.5-0.5*np.cos(2*np.pi*x/len(N))

N = 100
rw = np.cumsum(np.random.randn(N))
