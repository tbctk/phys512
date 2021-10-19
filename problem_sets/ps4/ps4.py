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

def get_derivs(pars,lmax=3000):
    # Simple 2-point derivative
    pars = np.array(pars)
    derivs = []
    for i in range(len(pars)):
        #print("*** Calculating derivatives of parameter ",i)
        p = pars[i]
        dp_i = p*0.1
        plus = np.concatenate((pars[:i],np.array([pars[i]+dp_i]),pars[i+1:]))
        f1 = get_spectrum(plus,lmax=lmax)
        minus = np.concatenate((pars[:i],np.array([pars[i]-dp_i]),pars[i+1:]))
        f2 = get_spectrum(minus,lmax=lmax)
        deriv_i = (f1-f2)/(2*dp_i)
        derivs.append(deriv_i)
    return np.array(derivs).T

## LM (Q2)

def get_model_derivs(m,lmax):
    model = get_spectrum(m,lmax=lmax)[:lmax]
    derivs = get_derivs(m,lmax=lmax)[:lmax]
    return model,derivs
        

def update_lam(lam,success):
    if success:
        lam = lam/1.5
        if lam < 0.5:
            lam = 0
    else:
        if lam == 0:
            lam = 1
        else:
            lam = lam*1.5**2
    return lam

def get_matrices(m,x,y,n_inv):
    lmax = len(x)
    model,derivs = get_model_derivs(m,lmax)
    r = y-model
    lhs = derivs.T@n_inv@derivs
    rhs = derivs.T@(n_inv@r)
    chisq = r.T@n_inv@r
    return chisq,lhs,rhs

def linv(mat,lam):
    mat = mat+lam*np.diag(np.diag(mat))
    return np.linalg.inv(mat)

def fit_lm(m,x,y,errs,niter=10,chitol=0.01):
    t1 = time.time()
    lam = 0
    lmax = len(y)
    n_inv = np.diag(errs**-1)
    chisq,lhs,rhs = get_matrices(m,x,y,n_inv)
    for i in range(niter):
        lhs_inv = linv(lhs,lam)
        dm = lhs_inv@rhs
        m_new = m+dm
        chisq_new,lhs_new,rhs_new = get_matrices(m_new,x,y,n_inv)
        if (chisq_new<chisq):
            if lam==0:
                if np.abs(chisq_new-chisq)<chitol:
                    print("Converged after ",i," iterations")
                    return m_new
            chisq = chisq_new
            lhs = lhs_new
            rhs = rhs_new
            m = m_new
            lam = update_lam(lam,True)
        else:
            lam = update_lam(lam,False)
        print("On iteration ",i," chisq is ",chisq," with m ",m)
    t2 = time.time()
    print("Took time ",t2-t1)
    return m,lhs


## MCMC (Q3)

def get_chisq(pars,x,y,noise):
    lmax = len(y)
    pred = get_spectrum(pars,lmax=lmax)[:lmax]
    return np.sum(((y-pred)/noise)**2)

def write_down(filename,arr,create=False,fmt='%.18e'):
    mode = 'wb' if create else 'ab'
    f = open(filename,mode)
    np.savetxt(f,[arr],fmt=fmt)
    f.close()

def prior_chisq(pars,par_priors,par_errs):
    if par_priors is None:
        return 0
    par_shifts = pars-par_priors
    return np.sum((par_shifts/par_errs)**2)

def mcmc(pars,step_size,x,y,noise,fname,nstep=1000,par_priors=None,par_errs=None):
    chisq = get_chisq(pars,x,y,noise)+prior_chisq(pars,par_priors,par_errs)
    npar = len(pars)
    chain = np.zeros([nstep,npar])
    chivec = np.zeros(nstep)
    labels = ["chisq","H0","ombh2","omch2","tau","As","ns"]
    write_down(fname,labels,create=True,fmt='%s')
    for i in range(nstep):
        trial_pars = pars+step_size*np.random.randn(npar)
        print("Trying pars ",trial_pars,"...")
        trial_chisq = get_chisq(trial_pars,x,y,noise)+prior_chisq(trial_pars,par_priors,par_errs)
        delta_chisq = trial_chisq-chisq
        accept_prob = np.exp(-0.5*delta_chisq)
        accept = np.random.rand(1)<accept_prob
        print("Trial chisq: ",trial_chisq," Accepted: ",accept)
        if accept:
            pars = trial_pars
            chisq = trial_chisq
        chain[i,:] = pars
        chivec[i] = chisq
        aline = np.concatenate(([chisq],pars))
        write_down(fname,aline)
    return chain,chivec

## IMPORTANCE SAMPLING (Q4)


# Initial parameter guess
pars = np.asarray([69,0.022,0.12,0.06,2.1e-9,0.95])
# Load data
data = np.loadtxt("COM_PowerSpect_CMB-TT-full_R3.01.txt").T
x = data[0]
y = data[1]
errs = 0.5*(data[2]+data[3])

def main1():
    # LM fit (question 2)
    pars_lm,curvature = fit_lm(pars,x,y,errs)
    cov = np.linalg.inv(curvature)
    p_errs_lm = np.diag(cov)
    np.savetxt("planck_fit_params.txt",np.asarray([pars_lm,p_errs_lm]).T)

def main2():
    # MCMC fit (question 3)
    pfm = np.loadtxt("planck_fit_params.txt").T
    pars_lm = pfm[0]
    p_errs_lm = pfm[1]
    chain,chivec = mcmc(pars_lm,10*p_errs_lm,x,y,errs,"planck_chain.txt")
    #np.savetxt("planck_chain_after.txt",np.asarray([chivec,chain]).T)
    pars_mcmc = np.mean(chain,axis=0)
    errs_mcmc = np.std(chain,axis=0)
    dark_energy_density = 1 - 10000*(pars_mcmc[1]+pars_mcmc[2])*pars_mcmc[0]**-2
    fm0 = 2e4*(pars_mcmc[1]+pars_mcmc[2])/pars_mcmc[0]**3
    fm1 = -1e4/pars_mcmc[0]**2
    fm2 = fm1
    ded_err = ((fm0*mcp[1,0])**2 + (fm1*mcp[1,1])**2 + (fm2*mcp[1,2])**2)**0.5
    np.savetxt("mcmc_params.txt",np.asarray([pars_mcmc,errs_mcmc]).T)
    np.savetxt("dark_energy_density.txt",[dark_energy_density,ded_err])

def main3():
    tau_prior = 0.054
    tau_err = 0.0074
    par_priors = np.zeros(len(pars))
    # Avoiding divide by zero errors
    par_errs = par_priors + 1e20
    par_priors[3] = tau_prior
    par_errs[3] = tau_err
    
    ## Importance sampling chain from q3
    q3_data = np.loadtxt("planck_chain.txt",skiprows=1)
    chain_q3 = q3_data[:,1:]
    nsamp = chain_q3.shape[0]
    weight = np.zeros(nsamp)
    chivec = weight
    for i in range(nsamp):
        chisq = prior_chisq(chain_q3[i,:],par_priors,par_errs)
        chivec[i] = chisq
    chivec = chivec-chivec.mean()
    weight = np.exp(0.5*chivec)
    w_sum = np.sum(weight)
    npar = chain_q3.shape[1]
    pars_importance = np.empty(npar)
    errs_importance = np.empty(npar)
    for i in range(npar):
        pars_importance[i] = np.sum(weight*chain_q3[:,i])/w_sum
        errs_importance[i] = np.sqrt(np.sum(weight*(chain_q3[:,i]-pars_importance[i])**2)/w_sum)
    np.savetxt("importance_params.txt",np.asarray([pars_importance,errs_importance]).T)
        
    ## MCMC    
    pars_mcmc = np.loadtxt("mcmc_params.txt")
    errs_im = np.loadtxt("importance_params.txt")[:,1]
    pars_lm = np.loadtxt("planck_fit_params.txt")[:,0]
    chain2,chivec2 = mcmc(pars_lm,3*errs_im,x,y,errs,"planck_chain_tauprior.txt",par_priors=par_priors,par_errs=par_errs)
    pars_tauprior = np.mean(chain2[:,1:],axis=0)
    np.savetxt("tauprior_params.txt",pars_tauprior)

def main():
    main1()
    main2()
    main3()
