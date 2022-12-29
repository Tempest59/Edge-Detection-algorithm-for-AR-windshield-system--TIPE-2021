# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:57:34 2021

@author: titit
"""
import pylab as plt
import numpy as np
from math import *
def distance_image(f,prec):
    s = np.linspace(0, f-prec, 1000)
    z = (s*f)/(s-f)
    plt.plot(s, -z, label='Courbe')  # Tracé de la courbe 3D
    plt.title("Distance image")
    plt.show()
    
def distance_objet(im,f,prec):
    z=-im
    return round((z*f)/(z-f),-int(round(log(prec))))

distance_image(10, 5)

# Plots the distance of virtual image and virtual object according to the optical system's parameters.
    