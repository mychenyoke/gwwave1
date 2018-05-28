import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz

sigma=10
sampling_rate=100
sin_omega=3

def gaussian(VALUE_SIN,x,mu,sig):
    return VALUE_SIN * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def sinfunction(A):
    x=np.arange(0,240)
    VALUE_SIN=A* np.sin(x * sin_omega)
    return VALUE_SIN

def gaussion_sin_function(Amplitude):
    a=sinfunction(Amplitude)
    z2=gaussian(a,time,0,sigma)
    return z2

time=np.linspace(-2,2,240)
template= gaussion_sin_function(2) #input gaussion_sin_function(Amplitude) 

#Here we do the fourier transform with template(gaussion_sin_function) 
frequency=np.fft.fftfreq(len(template),1/sampling_rate)
fft_template=np.fft.fft(template)

# Calculate the power_spectral_density

template_psd_quarter_sampling_rate,template_frequence_quarter_sampling_rate=mlab.psd(template,Fs=sampling_rate,NFFT=sampling_rate//4)
Amplitude_psd_quarter_sampling_rate=np.sqrt( template_psd_quarter_sampling_rate)

template_psd_half_sampling_rate,template_frequence_half_sampling_rate=mlab.psd(template,Fs=sampling_rate,NFFT=sampling_rate//2)
Amplitude_psd_half_sampling_rate=np.sqrt( template_psd_half_sampling_rate)

template_psd_sampling_rate,template_frequene_sampling_rate=mlab.psd(template,Fs=sampling_rate,NFFT=sampling_rate)
Amplitude_psd_sampling_rate=np.sqrt( template_psd_sampling_rate)

template_psd_twice_sampling_rate,template_frequence_twice_sampling_rate=mlab.psd(template,Fs=sampling_rate,NFFT=sampling_rate*2) 
Amplitude_psd__twice_sampling_rate=np.sqrt( template_psd_twice_sampling_rate)

template_psd_templetelen_sampling_rate,template_frequencyd_templetelen_sampling_rate=mlab.psd(template,Fs=sampling_rate,NFFT=len(template))  
Amplitude_psd_templetelen_sampling_rate=np.sqrt( template_psd_templetelen_sampling_rate)

plt.Figure(figsize=(50,30))
plt.subplots_adjust(left=0.2,bottom=0.2,top=0.9,right=0.9,hspace=1)

plt.subplot(2,1,1)
plt.plot(time,template)
plt.title("time_strain")
plt.xlabel("time")
plt.ylabel("strain")

plt.subplot(2,1,2)
plt.loglog(template_psd_quarter_sampling_rate,Amplitude_psd_quarter_sampling_rate,label="quarter_sampling_rate")
plt.loglog(template_psd_half_sampling_rate,Amplitude_psd_half_sampling_rate,label="half_sampling_rate")
plt.loglog(template_psd_sampling_rate,Amplitude_psd_sampling_rate,label="sampling_rate")
plt.loglog(template_psd_twice_sampling_rate, Amplitude_psd__twice_sampling_rate,label="twice_sampling_rate")
plt.loglog(template_psd_templetelen_sampling_rate,Amplitude_psd_templetelen_sampling_rate,label="templetelen_sampling_rate")
plt.title("Not_windowing_amplitude_spectral_density")    
plt.xlabel("frequency")
plt.ylabel("ASD")
    
plt.savefig("power_spectral_density.png")    