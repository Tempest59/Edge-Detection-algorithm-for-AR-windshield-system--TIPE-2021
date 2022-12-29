import cv2 as cv
import numpy as np
import pylab as plt

img = cv.imread('')

rows,cols,ch = img.shape
pts1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
pts2 = np.float32([[100,0],[cols-100,0],[0,rows],[cols,rows]])
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(1280,960))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

def déformation(img,offset,move_up_down):
    rows,cols,ch = img.shape
    
    if move_up_down == 0:
        pts1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
        pts2 = np.float32([[offset,0],[cols-offset,0],[0,rows],[cols,rows]])
    else:
        pts1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
        pts2 = np.float32([[0,0],[cols,0],[offset,rows],[cols-offset,rows]])
        
    M = cv.getPerspectiveTransform(pts1,pts2)
    
    return cv.warpPerspective(img,M,(1280,960))

def position(img):
    rows,cols = img.shape
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv.warpAffine(img,M,(cols,rows))
    return dst

# Warps the perspective of image to compensate for inclination of source screen.