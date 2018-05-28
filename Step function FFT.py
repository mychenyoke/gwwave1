import numpy as np
import matplotlib.pyplot as plt


#Prepare data
N=500
N0=1000
X=np.ones(N0)
X[:N]=np.zeros(N)
Time=np.arange(0,1000)

plt.figure(figsize=(10,16))
#Doing  the fourier transform
plt.subplot(3,1,1)
Nt=len(X)
dt=1/4096
plt.plot(Time,X)
plt.title("time verse strain")
plt.xlabel("time")
plt.ylabel("strain")

#calculate the sample frequency
frequence=np.fft.fftfreq(Nt,dt)
#print(frequence,len(frequence))
#calculate the fast-Fourier transform
hf=np.fft.fft(X)
# caculate the inverse Fast-Fourier transform
hff=np.fft.ifft(hf)
plt.subplot(3,1,2)
plt.plot(frequence,hf)
plt.plot(frequence,abs(hf))
plt.subplot(3,1,3)
plt.plot(Time,hff)
#print(X)