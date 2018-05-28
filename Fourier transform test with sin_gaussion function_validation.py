import matplotlib.pyplot as plt
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
#Do the inverse fourier transform to  verify  the wave
reverse_ensure=np.fft.ifft(fft_template)

#Do the match filtering with wave and window

#plt figure
plt.Figure(figsize=(50,30))
plt.subplots_adjust(left=0.2,bottom=0.1,top=0.9,right=1,hspace=1)

plt.subplot(3,1,1)
plt.title("time-strain")
plt.plot(time,template)
plt.xlabel("time")
plt.ylabel("strain")


plt.subplot(3,1,2)
plt.title("FFT_Singaussion_function_without_windowfunction")
plt.plot(frequency,fft_template)
plt.xlabel("frequency")
plt.ylabel("strain")


plt.subplot(3,1,3)
plt.title("time-strain_validation")
plt.plot(time,template)
plt.plot(time,reverse_ensure)
plt.xlabel("time")
plt.ylabel("strain")

plt.savefig("Fourier_transform_SinGaussion_validation")




