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

# Calculating the power_spectral_density with window

NFFT=sampling_rate//2
template_psd,template_freqs=mlab.psd(template,Fs=sampling_rate,NFFT=NFFT,noverlap=NFFT/8,window=signal.tukey(NFFT,alpha=0.1))

## calculateing time-domain signal calculate sin-gaussion function

dwindow=signal.tukey(len(template),alpha=0.9)
signal_with_window_function_time_space=template*dwindow

### using FFT calculate sin-gaussion function
f_template_fft=np.fft.fftfreq(len(signal_with_window_function_time_space),dt)
template_fft=np.fft.fft(template)*dt
signal_fft=np.fft.fft(signal_with_window_function_time_space)*dt



plt.plot(time,template,"r")
plt.xlabel("time")
plt.ylabel("strain")
plt.plot(time,signal_with_window_function_time_space,"g")
plt.title("The Gaussian_sin function  compare with window or not")
plt.savefig("The Gaussian_sin function compare with window or not")


