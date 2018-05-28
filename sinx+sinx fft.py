import numpy as np
import matplotlib.pyplot as plt

omega1=0.1
omega2=0.2
sample_rate=20

a=np.arange(0,100)
sina=np.sin(omega1*a)
sinb=np.sin(omega2*a)+np.sin(omega1*a)

plt.figure(figsize=(10,24))

plt.subplot(4,1,1)
plt.title("sinax")
plt.plot(a,sina)
plt.savefig("sinax")

plt.subplot(4,1,2)
plt.title("sinax+sinbx")
plt.plot(a,sinb)
plt.savefig("sinax+sinbx")

aa=[]
fft_frequency=np.fft.fftfreq(len(a),1/sample_rate)
fft_sina=np.fft.fft(sina)
#print(abs(fft_sina))
aa=abs(fft_sina)
for ab in aa:
    print(ab)

fft_sinb=np.fft.fft(sinb)

plt.subplot(4,1,3)
plt.title("FFT_Frequency_sinax")
plt.plot(fft_frequency,abs(fft_sina))
plt.savefig("FFT_Frequency_sinax")

plt.subplot(4,1,4)
plt.title("FFT_Frequency_sinax+sinbx")
plt.plot(fft_frequency,fft_sinb)
plt.savefig("FFT_Frequency_sinax+sinbx")