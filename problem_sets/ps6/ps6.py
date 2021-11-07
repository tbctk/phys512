import numpy as np
import json
from matplotlib import pyplot as plt
import simple_read_ligo as srl


## PRELIMINARIES

dirname = "LIGO/LOSC_Event_tutorial/"
f = open(dirname+"BBH_events_v3.json")
file_dict = json.load(f)
f.close()
name1 = "GW150914"
name2 = "LVT151012"
name3 = "GW151226"
name4 = "GW170104"
names = [name1,name2,name3,name4]

strains = [[],[]]
dts = [[],[]]
templates = [[],[]]

for i, name in enumerate(names):
    # File names
    tem = dirname+file_dict[name]["fn_template"]
    hav = dirname+file_dict[name]["fn_H1"]
    liv = dirname+file_dict[name]["fn_L1"]
        
    # Read data
    strain_h,dt_h,utc_h = srl.read_file(hav)
    strain_l,dt_l,utc_l = srl.read_file(liv)
    
    strains[0].append(strain_h)
    strains[1].append(strain_l)
    dts[0].append(dt_h)
    dts[1].append(dt_l)

    # Load and store templates
    template_h,template_l = srl.read_template(tem)
    templates[0].append(template_h)
    templates[1].append(template_l)

strains = np.asarray(strains)
dts = np.asarray(dts)
dt = dts[0,0]
assert all(dts[0]==dt) and all(dts[1]==dt)
fs = 1/dt
templates = np.asarray(templates)


def tukey_window(n,m):
    x = np.linspace(-np.pi,np.pi,m)
    ends = 0.5*(1+np.cos(x))
    mid = m//2
    win = np.ones(n)
    win[:mid] = ends[:mid]
    win[-mid:] = ends[-mid:]
    return win

npt = len(strains[0][0])
win = tukey_window(npt,npt//5)
showfigs = True

### PART A

def get_noise_model(strains,win=None):
    (ns,npt) = strains.shape
    strains_ps = []
    if win is None:
        for strain in strains:
            strains_ps.append(np.fft.rfft(strain)**2)
    else:
        for strain in strains:
            strains_ps.append(np.fft.rfft(strain*win)**2)
    # Use the average of these strain power spectra as the noise model.
    strains_ps = np.asarray(strains_ps)
    noise_model_ps = np.sum(strains_ps,axis=0)/ns
    n_smooth = 6
    noise_model_ps = np.convolve(noise_model_ps,np.ones(n_smooth)/n_smooth,mode='same')
    return np.abs(noise_model_ps)

noise_model_h = get_noise_model(strains[0],win=win)
noise_model_l = get_noise_model(strains[1],win=win)
noise_models = np.asarray([noise_model_h,noise_model_l])

if showfigs:
    freq = np.fft.rfftfreq(len(strains[0,0]),d=dt)
    plt.figure()
    plt.loglog(freq,noise_models[0],label="Noise Model for Hanford Detector")
    plt.loglog(freq,noise_models[1],label="Noise Model for Livingston Detector")
    plt.legend()
    plt.show()

### PART B

def whiten(data,noise_ps,dt,win=None):
    if win is None:
        data_ft = np.fft.rfft(data)
    else:
        data_ft = np.fft.rfft(data*win)
    white_data_ft = data_ft/np.sqrt(noise_ps)
    norm = 1/np.sqrt(1/(2*dt))
    white_data_ft = white_data_ft*norm
    white_data = np.fft.irfft(white_data_ft)
    return white_data

def matched_filter(data,template,win=None):
    freq = np.fft.rfftfreq(len(template),d=dt)
    df = np.abs(freq[1]-freq[0])
    if win is None:
        data_ft = np.fft.rfft(data)/fs
        template_ft = np.fft.rfft(template)/fs
    else:
        data_ft = np.fft.rfft(data*win)/fs
        template_ft = np.fft.rfft(template*win)/fs
    return np.fft.irfft(data_ft*np.conj(template_ft))

whitened_strains = np.empty(strains.shape)
whitened_templates = np.empty(templates.shape)
correlations = [[],[]]

for i in range(2):
    for j in range(len(names)):
        whitened_strains[i,j] = whiten(strains[i,j],noise_models[i],dts[i,j],win=win)
        whitened_templates[i,j] = whiten(templates[i,j],noise_models[i],dts[i,j],win=win)
        correlations[i].append(matched_filter(whitened_strains[i,j],whitened_templates[i,j],win=win))

correlations = np.asarray(correlations)

if showfigs:
    n = len(correlations[0,0])
    t = np.linspace(0,n*dt,n)
    detectors = ["Hanford","Livingston"]
    fig,axs = plt.subplots(4,2)
    for i in range(2):
        for j in range(len(names)):
            y = np.fft.fftshift(correlations[i,j])
            axs[j,i].plot(t,y)
            axs[j,i].set_title(detectors[i]+" Event "+str(j+1),fontsize=8)
            #axs[j,i].plot(
            plt.show()

### PART C

# Estimate noise based on matched filter output
#noise_ests = np.empty(strains.shape)
snrs = np.empty((2,4))
for i in range(2):
    for j in range(4):
        ne = np.std(correlations[i,j][-50000:])
        #noise_ests[i,j] = ne
        snrs[i,j] = np.max(np.abs(correlations[i,j]))/ne

if False:

    N_dict = {}

    for name in names:
        N_dict[name] = {}

    for name in names:
        # File names
        f_tem = dirname+file_dict[name]["fn_template"]
        f_hav = dirname+file_dict[name]["fn_H1"]
        f_liv = dirname+file_dict[name]["fn_L1"]
        
        # Read data
        strain_h,dt_h,utc_h = srl.read_file(f_hav)
        strain_l,dt_l,utc_l = srl.read_file(f_liv)
        
        assert len(strain_h)==len(strain_l)
        
        # Choose a Tukey window function that has a flat portion in the middle
        n = len(strain_h)
        win = tukey_window(n,n//4)

        # Fourier transforms of the strains
        strain_ft_h = np.fft.rfft(win*strain_h)
        strain_ft_l = np.fft.rfft(win*strain_l)
        
        # Use the power spectra of the strain as the starting estimate of the 
        # noise model since we know the data is dominated by noise.
        noise_ps_h = np.abs(strain_ft_h)**2
        noise_ps_l = np.abs(strain_ft_l)**2

        # Smoothening the power spectrum of the noise 
        for i in range(10):
            noise_ps_h=(noise_ps_h+np.roll(noise_ps_h,1)+np.roll(noise_ps_h,-1))/3
            noise_ps_l=(noise_ps_l+np.roll(noise_ps_l,1)+np.roll(noise_ps_l,-1))/3

        N_dict[name]["noise_ps_h"] = noise_ps_h
        N_dict[name]["noise_ps_l"] = noise_ps_l
        
        ### PART B

        # Load and store templates
        template_h,template_l = srl.read_template(f_tem)
        N_dict[name]["template_h"] = template_h
        N_dict[name]["template_l"] = template_l

        # Pre-whitening
        strain_ft_h_white = strain_ft_h/np.sqrt(noise_ps_h)
        template_ft_h_white = np.fft.rfft(t_h*win)/np.sqrt(noise_ps_h)
        t_white = np.fft.irfft(tft_h_white)
        

        xcorr = np.fft.irfft(sft_h_white*np.conj(tft_white))
        N_dict[name]["xcorr"] = xcorr
