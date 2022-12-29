"""
0. import
"""

import math as m
import numpy as np
import matplotlib.pyplot as plt
import cv2 

def rescale_line(array, len2):
    nR0 = len(array)     # source number of rows 
    r = int(nR0/len2)
    return [ array[int(nR0 * r / len2)] for r in range(int(len2))]

def scale2(im, ratio):
    nR0 = len(im[0])     # source number of rows 
    nC0 = len(im)  # source number of columns 
    nR = nR0 * ratio
    nC = nC0 * ratio
    a = []
    for i in range(1,nC0,2):
            a.append(rescale_line(im[i],nR0*ratio))
    return a
"""
1. Low-pass filter: Removes noise from input frame.
"""


def noyau_gaussien(sigma, diam): 

    """
    Gaussian filter's kernel
    
    Parameters
    ----------
    sigma : float
        Filter's σ value.
    
    size : int
        Size of the kernel.
    
    
        
    returns
    -------
    ng : numpy.ndarray
        Gaussian filter's kernel.
    """
    
    Rnoy = diam // 2 
    
    coef = 1 / ( 2 * m.pi * (sigma**2) )
    
    y, x = np.mgrid[ -Rnoy : Rnoy+1, -Rnoy : Rnoy+1 ]
    
    ng =  np.exp( -1 * ((x**2 + y**2) / (2 * (sigma**2)))) * coef
    
    return ng




def convolution(img, noy):
    
    """
    Convolutionnal product from kernel.
    
    Parameters
    ----------
    img : array
        2D array of the image.
    
    noy : array
        Kernel's array.
    
    
        
    Returns
    -------
    imgConv : array
        Filtered image.
    """
    
    H, L = img.shape[:2]
    
    Dnoy = noy.shape[0]
    Rnoy = Dnoy // 2
    
    noyBonneDim = np.reshape(noy, (Dnoy, Dnoy,1))
                             
    imgBord = np.pad(img, ((Rnoy, Rnoy), (Rnoy, Rnoy), (0, 0)), 'edge' )
    
    imgConv = np.zeros_like(img)
    
    for i in range( H ):
        
        for j in range( L ):
            
            
            imgConv[i, j] \
            = np.sum(imgBord[ i: i + Dnoy, j: j + Dnoy ] * noyBonneDim ,\
                     axis = (0, 1))

    return imgConv


    

def nb(img):

    
    """
    Returns B&W version of the frame.
    
    Parameters
    ----------
    img : array
       2D array of the image.
    
    
    
    Returns
    -------
    : array
        B&W image gradient's array.   
    """
    
    return np.mean( img[:, :], axis = 2).astype(np.int32)

def nb2(img):
    c = np.shape(img)[0]
    l = np.shape(img)[1]
    a = np.empty((c,l))
    for i in range(c):
        for j in range(l):
            a[i][j] = np.mean(img[i][j])
    return a
            


def gradient(imgNB):
    
    """
    Computation of B&W image's gradient.
    
    Parameters
    ----------
    img : array
        2D array of the image.
    
    
        
    REturns
    -------
    gradtheta : array
        Gradient's array and gradient's angles array.
    """
    
    H, L = imgNB.shape[:2]
    
    gradTheta = np.full((H,L,2), [0.,2])
    
    convolution = np.array([-1,1])
    
    imgBord = np.pad(imgNB, ((0, 1), (0, 1)), 'edge')
    
    for i in range(H):
        
        for j in range(L):
            
            """
            Computation of the gradient.
            """
            
            dp_img_i = np.sum(imgBord[ i: i+2, j ] * convolution)
            
            dp_img_j = np.sum(imgBord[ i, j: j+2 ] * convolution)
            
            gradTheta[i,j,0] = m.sqrt(dp_img_i**2 + dp_img_j**2)
            
            """ 
            Gradient's direction computation.
            """
            
            if dp_img_j != 0:
                
                gradTheta[i,j,1] = m.atan(dp_img_i / dp_img_j ) // (m.pi/4)
    
    return gradTheta




def maximum_local(grad):
    """
    Returns local maximum of the gradient;
    
    Parameters
    ----------
    grad : array
        2D array of the gradient.
    
    
        
    Returns
    -------
    gradMax : array
        Gradient's local max array.
    """
    
    H,L = grad.shape[:2]
    
    gradBord = np.pad(grad[:,:,0], ((1, 1), (1, 1)), 'edge')
    
    gradMax = grad[:,:,0].copy()
    
    coordAngle = [3, 2   ,  1,\
                  4, None,  0,\
                 -3, -2  , -1]
                    
                    
    for i in range(H):
        
        for j in range(L):
            
            k = 0
            
            voisins = np.reshape( gradBord[i: i+3, j: j+3], 9)
            
            while grad[i,j,1] != coordAngle[k]:
                
                k += 1
            
            if grad[i,j,0] <= voisins[k] or grad[i,j,0] <= voisins[-k-1]:
                
                gradMax[i,j] = 0
                
    return gradMax


def hysterisis(gradMax,tauh,tau1): # à refaire
    
    """
    Returns the edges verifying the thresholds tauh and tau1 of gradient's norms.
    
    Parameters
    ----------
    gradMax : array
        2D array of the gradient.
        
    tauh : float
    
    tau1 : float
    
    
        
    Returns
    -------
    hyst : array
        Gradient's local max array.
    """
    
    H, L = gradMax.shape[:2]
    
    seuil = np.zeros_like(gradMax, dtype = float)
    
    hyst = np.zeros_like(gradMax, dtype = int)
    
    for i in range(H):
        
        for j in range(L):
            
            if gradMax[i, j] > tauh:
                
                seuil[i, j] = 1
                
            elif gradMax[i, j] > tau1:
                
                seuil[i, j] = 0.5
                
    seuil = np.pad(seuil, ((1, 1), (1, 1)), 'edge')
                
    for i in range(H):
        
        for j in range(L):
            
            if seuil[i+1, j+1] == 1:
                hyst[i: i+3, j: j+3] \
                += ( seuil[i: i+3, j: j+3] >= 0.5 ).astype(int)
                
                
            
    return (hyst > 1 ).astype(int)


def contour(img, sigma, diam, tauh, tau1):
    
    return hysterisis(maximum_local(gradient(nb(convolution(cv2.resize(img,(1280,720)), noyau_gaussien(sigma, diam))))), tauh, tau1)

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True: 
    ret, frame = vid.read()
    contour = contour(frame, .8,3,20,5)
    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
    
    
        
