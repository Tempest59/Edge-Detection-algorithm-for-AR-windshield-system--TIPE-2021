# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:48:36 2021

@author: titit
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  # Fonction pour la 3D
import numpy as np


f = np.linspace(1,50,1000) 
s = np.linspace(1,50, 1000) # Création du tableau de l'axe y 
z = (s*f)/(s-f) 
# Tracé du résultat en 3D
fig = plt.figure()
ax = fig.gca(projection='3d')  # Affichage en 3D
ax.plot(s, f, z, label='Courbe')  # Tracé de la courbe 3D
plt.title("Distance image")
ax.set_xlabel('Distance objet')
ax.set_ylabel('Focale lentille')
ax.set_zlabel('Distance image')
plt.tight_layout()
plt.show()

def distance_image(f):
    s = np.linspace(f+1, 50, 1000)
    z = (s*f)/(s-f)
    plt.plot(s, -z, label='Courbe')  # Tracé de la courbe 3D
    plt.title("Distance image")
    ax.set_xlabel('Distance 3Dobjet')
    ax.set_ylabel('Focale lentille')
    ax.set_zlabel('Distance image')
    plt.show()
    
    
    