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

#using turkey window

dwindow_turkey=signal.tukey(len(template),alpha=0.1)
fft_match_filtering_sampling_turkey=np.fft.fftfreq(len(dwindow_turkey),1/sampling_rate)
fft_match_filtering_turkey=np.fft.fft(template*dwindow_turkey)*(1/sampling_rate)

#using hanning as window function

dwindow_hanning=np.hanning(len(template))
fft_match_filtering_sampling_hanning=np.fft.fftfreq(len(dwindow_hanning),1/sampling_rate)
fft_match_filtering_hanning=np.fft.fft(template*dwindow_hanning)*(1/sampling_rate)


#using blackman as window function

dwindow_blackman=signal.blackman(len(template))
fft_match_filtering_sampling_hanning=np.fft.fftfreq(len(dwindow_blackman),1/sampling_rate)
fft_match_filtering_blackman=np.fft.fft(template*dwindow_blackman)*(1/sampling_rate)

#plt figure
plt.Figure(figsize=(60,30))
plt.subplots_adjust(left=0.0,bottom=0.1,top=0.9,right=1,hspace=2)


plt.subplot(4,1,1)
plt.title("FFT_Singaussion_function_no_window")
plt.plot(frequency,fft_template)
plt.xlabel("frequency")
plt.ylabel("strain")


plt.subplot(4,1,2)
plt.title("FFT_Singaussion_function_with_windowfunction_turkey")
plt.plot(fft_match_filtering_sampling_turkey,fft_match_filtering_turkey)
plt.xlabel("frequency")
plt.ylabel("strain")

plt.subplot(4,1,3)
plt.title("FFT_Singaussion_function_with_windowfunction_hanning")
plt.plot(fft_match_filtering_sampling_hanning,fft_match_filtering_hanning)
plt.xlabel("frequency")
plt.ylabel("strain")

plt.subplot(4,1,4)
plt.title("FFT_Singaussion_function_with_windowfunction_blackman")
plt.plot(fft_match_filtering_sampling_hanning,fft_match_filtering_blackman)
plt.xlabel("frequency")
plt.ylabel("strain")

plt.savefig("Fourier_transform_SinGaussion_acting_with_window_function")