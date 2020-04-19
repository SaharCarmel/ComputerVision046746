###############################################################################
#                            pkg imports                                      #
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


###############################################################################
#                            fun imports                                      #
###############################################################################
from scipy.signal import argrelextrema


###############################################################################
#                            constants                                        #
###############################################################################
sigma0  = 1
k       = np.sqrt(2)
levels  = [-1, 0, 1, 2, 3, 4]
thetaC  = 0.03
thetaR  = 12


###############################################################################
#                            functions                                        #
###############################################################################

###############################################################################
def loadImage(fp):
###############################################################################
    im = cv2.imread(fp)
    #plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    #_ = plt.axis('off')
    #plt.show()
    return im


###############################################################################
def createGaussianPyramid(im, sigma0, k, levels):
###############################################################################
    GaussianPyramid = []

    for i in range(len(levels)):
        sigma_ = sigma0 * k ** levels[i]
        size = int(np.floor(3 * sigma_ * 2) + 1)
        blur = cv2.GaussianBlur(im, (size, size), sigma_)
        GaussianPyramid.append(blur)

    return np.stack(GaussianPyramid)


###############################################################################
def displayPyramid(pyramid):
###############################################################################
    plt.figure(figsize = (16, 5))
    plt.imshow(np.hstack(pyramid), cmap = 'gray')
    plt.axis('off')
    plt.show()

###############################################################################
def converToGrayNormalize(im):
###############################################################################
    norm_image = cv2.normalize(
        im, 
        None, 
        alpha = 0, 
        beta = 1,
        norm_type = cv2.NORM_MINMAX, 
        dtype = cv2.CV_32F
    )
    im_grayscale = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
    return im_grayscale

###############################################################################
def createDoGPyramid(GaussianPyramid, levels):
###############################################################################
    DoGPyramid = []

    for i, x in enumerate(GaussianPyramid):
        if i == 0:
            continue
        DoGPyramid.append(GaussianPyramid[i] - GaussianPyramid[i - 1])
    DoGLevels = levels

    return DoGPyramid, DoGLevels

###############################################################################
def computePrincipalCurvature(DoGPyramid):
###############################################################################
    PrincipalCurvature = []

    for _filter in DoGPyramid:
        _tempMat = np.zeros_like(_filter)
        Dxx, Dyy, Dxy = getSecondDerivative(_filter)
        for x, y in np.ndindex(Dxx.shape):
            hessian = np.array([
                [Dxx[x, y], Dxy[x, y]],
                [Dxy[x, y], Dyy[x, y]]
            ])
            _tempMat[x, y] = calculateCurvature(hessian)
        PrincipalCurvature.append(_tempMat)

    return PrincipalCurvature

###############################################################################
def getSecondDerivative(im):
###############################################################################
    Dxx = cv2.Sobel(im, cv2.CV_64F, dx = 2, dy = 0, ksize = 3)
    Dyy = cv2.Sobel(im, cv2.CV_64F, dx = 0, dy = 2, ksize = 3)
    Dxy = cv2.Sobel(im, cv2.CV_64F, dx = 1, dy = 1, ksize = 3)
    return Dxx, Dyy, Dxy

###############################################################################
def calculateCurvature(hessian):
###############################################################################
    tr = np.trace(hessian)
    det = np.linalg.det(hessian)
    if det == 0:
        det = np.finfo(float).eps
    R = tr ** 2 / det
    return R

###############################################################################
def getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature, 
    th_contrast, th_r):
###############################################################################
    extrema = []
    for level, im in enumerate(DoGPyramid):
        level_minima, level_maxima = findLevelExtrema(im)

        if level == 0:
            prevLevel = None
        else:
            prevLevel = DoGPyramid[level - 1]
        
        if level == len(DoGPyramid) - 1:
            nextLevel = None
        else:
            nextLevel = DoGPyramid[level + 1]
        
        if prevLevel is None and nextLevel is None:
            continue

        for pt in level_minima:
            if isSpatialExtrema(pt, prevLevel, im, nextLevel, 'min'):
                extrema.append(pt + (level,))
        for pt in level_maxima:
            if isSpatialExtrema(pt, prevLevel, im, nextLevel, 'max'):
                extrema.append(pt + (level,))

    locsDoG = []
    
    print('number of extrema before threshold: ' + str(len(extrema)))

    for pt in extrema:
        x, y, level = pt
        if np.abs(DoGPyramid[level][x, y]) > th_contrast and \
            PrincipalCurvature[level][x, y] < th_r:
            locsDoG.append(pt)

    print('number of extrema after threshold: ' + str(len(locsDoG))) 

    return locsDoG


###############################################################################
def findEightNeighbors(x, y):
###############################################################################
    p1 = (x - 1, y - 1) 
    p2 = (x, y - 1)
    p3 = (x + 1, y - 1)
    p4 = (x - 1, y)
    p5 = (x + 1, y)
    p6 = (x - 1, y + 1)
    p7 = (x, y + 1)
    p8 = (x + 1, y + 1)
    return p1, p2, p3, p4, p5, p6, p7, p8


###############################################################################
def findLevelExtrema(im):
###############################################################################
    max_x, max_y = im.shape
    max_x-=1
    max_y-=1
    max_pts = []
    min_pts = []

    for x, y in np.ndindex(im.shape):
        currIndex = (x, y)
        currVal = im[currIndex]
        currPixelCoords = findEightNeighbors(x, y)
        vals = []
        for p in currPixelCoords:
            x, y = p
            if x > max_x or x < 0 or y > max_y or y < 0:
                continue
            vals.append(im[p])

        max_val = max(vals)
        min_val = min(vals)
        if currVal > max_val:
            max_pts.append(currIndex)
            continue
        if currVal < min_val:
            min_pts.append(currIndex)

    return min_pts, max_pts

###############################################################################
def isSpatialExtrema(pt, prevLevel, currLevel, nextLevel, extremumType):
###############################################################################
    currVal = currLevel[pt]
    
    if prevLevel is None:
        if extremumType == 'min':
            prevVal = np.PINF
        else:
            prevVal = np.NINF
    else:
        prevVal = prevLevel[pt]

    if nextLevel is None:
        if extremumType == 'min':
            nextVal = np.PINF
        else:
            nextVal = np.NINF
    else:
        nextVal = nextLevel[pt]

    if extremumType == 'min' and currVal < prevVal and currVal < nextVal:
        return True
    elif extremumType == 'max' and currVal > prevVal and currVal > nextVal:
        return True

    return False

###############################################################################
def DoGdetector(im, sigma0, k, levels, th_contrast, th_r):
###############################################################################
    GaussianPyramid = createGaussianPyramid(im, 
        sigma0 = sigma0, 
        k = k, 
        levels = levels
    )
    DoGPyramid, DoGLevels = createDoGPyramid(GaussianPyramid, levels)
    curavaturePyramid = computePrincipalCurvature(DoGPyramid)
    locsDoG = getLocalExtrema(
        DoGPyramid, 
        DoGLevels, 
        curavaturePyramid, 
        th_contrast, 
        th_r
    )

    return locsDoG, GaussianPyramid


###############################################################################
#                           code running                                      #
###############################################################################

###############################################################################
def main():
###############################################################################
    # section 1.1 - load image
    fp_im = os.path.join(os.getcwd(), 'data','polygons.jpg')
    im = loadImage(fp_im)

    # section 1.2 - gaussian pyramid
    im_grayscale = converToGrayNormalize(im)
    GaussianPyramid = createGaussianPyramid(
        im_grayscale, 
        sigma0 = sigma0, 
        k = k, 
        levels = levels
    )
    #displayPyramid(GaussianPyramid)

    # section 1.3 - the dog pyramid
    DoGPyramid, DoGLevels = createDoGPyramid(GaussianPyramid, levels)
    #displayPyramid(DoGPyramid)

    # section 1.4 - edge supression
    curavaturePyramid = computePrincipalCurvature(DoGPyramid)

    # section 1.5 - detecting extrema
    locsDoG = getLocalExtrema(DoGPyramid, 
        DoGLevels, 
        curavaturePyramid,
        thetaC,
        thetaR
    )

    #print(locsDoG)

    plt.figure(figsize=(15,15))
    plt.imshow(im)
    x = []
    y = []
    for point in locsDoG:
        x.append(point[1])
        y.append(point[0])
    plt.scatter(x,y,marker='o', c='g')
    plt.show()


###############################################################################
def dog_detector_init():
###############################################################################
    fp_im = os.path.join(os.getcwd(), 'data','model_chickenbroth.jpg')
    im = loadImage(fp_im)
    im_grayscale = converToGrayNormalize(im)
    locsDoG, GaussianPyramid = DoGdetector(
        im_grayscale, 
        sigma0, 
        k, 
        levels, 
        thetaC, 
        thetaR
    )

if __name__ == "__main__":
    main()
    #dog_detector_init()