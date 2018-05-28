import matplotlib.pyplot as plt
import numpy as np

x=np.arange(0,360)
A=np.sin(x * np.pi/15.0)

# We used A as amplitude and means is zero,standard=c
def gaussian(A,x,mu,sig):
    return A* np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

z1=gaussian(1,np.linspace(-3,3,360),0,1)
z2=gaussian(A,np.linspace(-3,3,360),0,1)


plt.figure(figsize=(10,8))
#plt.plot(x,A)
#plt.plot(x,z1)
plt.plot(x,z2)
plt.xlabel("time")
plt.ylabel("strain")
plt.savefig("gaussionsin.png")
plt.show() 
plt.close