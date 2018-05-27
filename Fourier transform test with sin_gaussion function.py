import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from scipy import signal
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz

sigma=0.1
sampling_rate=0.1
sin_omega=1



def gaussian(VALUE_SIN,x,mu,sig):
    return VALUE_SIN * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def sinfunction(A):
    x=np.arange(0,360)
    VALUE_SIN=A* np.sin(x * sin_omega)
    return VALUE_SIN

def gaussion_sin_function(Amplitude):
    a=sinfunction(Amplitude)
    z2=gaussian(a,time,0,sigma)
    return z2

time=np.linspace(-3,3,360)
template= gaussion_sin_function(2) #input gaussion_sin_function(Amplitude) 

#Here we do the fourier transform with template(gaussion_sin_function) 
frequency=np.fft.fftfreq(len(template),1/sampling_rate)
fft_template=np.fft.fft(template)
#Do the inverse fourier transform to  verify  the wave
reverse_ensure=np.fft.ifft(fft_template)

#Do the match filtering with wave and window

#using turkey window
dwindow=signal.tukey(len(template),alpha=0.1)
fft_match_filtering_sampling=np.fft.fftfreq(len(dwindow),1/sampling_rate)
fft_match_filtering=np.fft.fft(template*dwindow)*(1/sampling_rate)



#plt figure
plt.Figure(figsize=(30,30))
plt.subplots_adjust(left=0.0,bottom=0.0,top=1,right=1,hspace=2)

plt.subplot(4,1,1)
plt.plot(time,template)
plt.xlabel("time")
plt.ylabel("strain")

plt.subplot(4,1,2)
plt.title("FFT_Singaussion_function_without_windowfunction")
plt.plot(frequency,fft_template)
plt.xlabel("frequency")
plt.ylabel("strain")


plt.subplot(4,1,3)
plt.plot(time,template)
plt.plot(time,reverse_ensure)
plt.xlabel("time")
plt.ylabel("strain")

plt.subplot(4,1,4)
plt.plot(fft_match_filtering_sampling,np.abs(fft_match_filtering))
plt.xlabel("frequency")
plt.ylabel("strain")
plt.title("FFT_Singaussion_function_with_windowfunction")