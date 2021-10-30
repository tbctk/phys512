import numpy as np
from matplotlib import pyplot as plt

def convolve(arr1,arr2):
    ft1 = np.fft.fft(arr1)
    ft2 = np.fft.fft(arr2)
    return np.abs(np.fft.ifft(ft1*ft2))

def gauss(pars,x):
    offset = pars[0]
    amp = pars[1]
    x0 = pars[2]
    sig = pars[3]
    return offset+amp*np.exp(-0.5*(x-x0)**2/sig**2)

## PROBLEM 1
def shift_array(arr,shift):
    assert type(shift) == int
    N = len(arr)
    arr2 = np.zeros(N)
    arr2[shift] = 1
    return np.abs(convolve(arr,arr2))

## PROBLEM 2
def correlation(arr1,arr2):
    dft1 = np.fft.fft(arr1)
    dft2 = np.fft.fft(arr2)
    return np.abs(np.fft.ifft(dft1*np.conj(dft2)))

## PROBLEM 3
def self_corr(arr,shift=0):
    arr2 = shift_array(arr,shift)
    return correlation(arr,arr2)

## PROBLEM 4
def conv_safe(arr1,arr2):
    n1 = len(arr1)
    n2 = len(arr2)
    n_safe = n1+n2
    arr1_safe = np.zeros(n_safe)
    arr2_safe = np.zeros(n_safe)
    arr1_safe = np.concatenate((np.zeros(n2),arr1))
    arr2_safe = np.concatenate((np.zeros(n1),arr2))
    return convolve(arr1_safe,arr2_safe)



## EXECUTION
savefigs = False

def setup():
    pars = [0,1,50,10]
    x = np.arange(100)
    global y
    y = gauss(pars,x)

def main1():
    setup()
    plt.figure()
    plt.plot(y)
    plt.plot(shift_array(y,int(len(y)/2)))
    plt.savefig("gauss_shift.png") if savefigs else None
    plt.show()

def main2():
    setup()
    plt.figure()
    plt.plot(correlation(y,y))
    plt.savefig("gauss_correlation.png") if savefigs else None
    plt.show()

def main3():
    setup()
    plt.figure()
    for shift in range(0,60,10):
        y_shifted = shift_array(y,shift)
        label = str(shift)
        plt.plot(correlation(y,y_shifted),label=label)
    plt.legend(loc="upper right",title="Shift")
    plt.savefig("gauss_correlation_shift.png") if savefigs else None
    plt.show()

def main4():
    pars = [0,1,50,10]
    x = np.arange(100)
    y = gauss(pars,x)
    pars2 = [0,1,20,4]
    x2 = np.arange(60)
    y2 = gauss(pars2,x2)
    conv = conv_safe(y,y2)
    plt.figure()
    plt.plot(conv)
    plt.savefig("safe_convolution.png") if savefigs else None
    print(len(conv))

