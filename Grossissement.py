

import pylab as plt
import numpy as np
import math

def distance_image(s,f):
    return (s*f)/(f-s)

def grossissement(s,s_p):
    return s_p/s

s1=np.linspace(15,19.9,50)
s2=np.linspace(20,29.8,50)
s3=np.linspace(40,49.5,50)

s_p1 = distance_image(s1,20)
g1 = grossissement(s1,s_p1)
s_p2 = distance_image(s2,30)
g2 = grossissement(s2,s_p2)
s_p3 = distance_image(s3,50)
g3 = grossissement(s3,s_p3)

plt.xlabel("Distance image")
plt.ylabel("Grossissement")
plt.plot(s_p1,g1,'r--',s_p2,g2,'b-',s_p3,g3,'g^')
plt.show()

# Plots the distance of the virtual image created by an optical system against the magnifying factor of the optical system.
