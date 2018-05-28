import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz

sigma=0.5
sampling_rate=100
dt=1/sampling_rate
sin_omega=2

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

#Generated The random 
noise=np.random.rand(len(template))

##FFT The generated noise

fft_noise_frequency=np.fft.fftfreq(len(noise),dt)
fft_noise=np.fft.fft(noise)*dt

###PSD
NFFT=len(template)//4
Random_noise_psd,f_random_noise=mlab.psd(noise,Fs=sampling_rate,NFFT=NFFT,window=signal.tukey(NFFT,alpha=0.1))

plt.loglog(f_random_noise,Random_noise_psd)
plt.loglog(fft_noise_frequency,fft_noise)