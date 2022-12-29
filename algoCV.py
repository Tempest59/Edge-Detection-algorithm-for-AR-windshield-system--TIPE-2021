import cv2
import cmapy # provides plt's colormaps to use with OpenCV
import numpy as np
import pylab as plt

def warp(img,offset,move_up_down,res): # Create perspective warping to compensate for screen inclination
    rows,cols,ch = img.shape
    
    if offset >= cols//2:
        return cv2.imread('monke.jpg')
    
    if move_up_down == 0:
        pts1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
        pts2 = np.float32([[offset,0],[cols-offset,0],[0,rows],[cols,rows]])
    else:
        pts1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
        pts2 = np.float32([[0,0],[cols,0],[offset,rows],[cols-offset,rows]])
        
    M = cv2.getPerspectiveTransform(pts1,pts2)
    
    return cv2.warpPerspective(img,M,res)

def position(img): # modify frame's position
    rows,cols,pix = img.shape
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True: 
    ret, frame = vid.read()
    Cannyframe = img_colorized = cv2.applyColorMap(cv2.Canny(frame,300,100), cmapy.cmap('hot')) # Edge detection using Canny's algorithm
    szCannyframe = cv2.resize(Cannyframe, (1280,720)) # Resize edge detected frame
    defCannyframe = position(warp(Cannyframe,0,1,(1920,1080))) # Warp edge detected frame
    
    cv2.imshow('video', defCannyframe)        
    if cv2.waitKey(4) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

