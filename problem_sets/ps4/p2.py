import numpy as np
import camb
import time

def get_spectrum(pars,lmax=3000):
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]
    spect = tt[2:]
    return tt[2:]

if False:
    # Speedy function for testing.
    # This method implements the function 
    def get_spectrum(pars,lmax=3000):
        x = np.arange(lmax)
        ans = x*0
        for i in range(len(pars)):
            ans = ans+pars[i]*x**i
        return ans
    # This method generates some random looking data 
    speedy_pars = [1,2,3,4,5,6]
    speedy_pars2 = [1.1,1.9,3.4,4.2,4.9,6.6]
    def speedy_data(size,pars=speedy_pars):
        x = np.arange(size)
        y = get_spectrum(pars,lmax=size)
        noise = np.random.randn(len(y))
        return x,y+noise

def get_derivs(pars,lmax=3000):
    # Simple 2-point derivative
    pars = np.array(pars)
    derivs = []
    for i in range(len(pars)):
        #print("*** Calculating derivatives of parameter ",i)
        p = pars[i]
        dp_i = p*0.01
        plus = np.concatenate((pars[:i],np.array([pars[i]+dp_i]),pars[i+1:]))
        f1 = get_spectrum(plus,lmax=lmax)
        minus = np.concatenate((pars[:i],np.array([pars[i]-dp_i]),pars[i+1:]))
        f2 = get_spectrum(minus,lmax=lmax)
        deriv_i = (f1-f2)/(2*dp_i)
        derivs.append(deriv_i)
    return np.array(derivs).T

def newton(m,l,y,niter=10):
    # ell is the first column of the data, just natural numbers
    # y is the second column, the actual spectral data
    t1 = time.time()
    lmax = len(y)
    m_0 = m
    for i in range(niter):
        model = get_spectrum(m,lmax=lmax)[:lmax]
        r = y-model
        derivs = get_derivs(m,lmax=lmax)[:lmax]
        lhs = derivs.T@derivs
        rhs = derivs.T@r
        dm = np.linalg.inv(lhs)@rhs
        while not m[3]+dm[3] > 0:
            dm[3] = dm[3]/2
        m = m+dm
        chisq = np.sum(r**2)
        print("On iteration ",i," chisq is ",chisq," with new parameters ",m)
    t2 = time.time()
    print("Newton's completed after time ",t2-t1)
    return m

pars = np.asarray([60,0.02,0.1,0.05,2e-9,1])
better_pars = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
data = np.loadtxt("COM_PowerSpect_CMB-TT-full_R3.01.txt").T#[:,:500]
