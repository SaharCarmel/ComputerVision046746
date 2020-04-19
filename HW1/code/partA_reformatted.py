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
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im / 255
    return im

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
    locsDoG = []

    max_x, max_y = DoGPyramid[0].shape
    max_x-=1
    max_y-=1

    for level, im in enumerate(DoGPyramid):
        mask_c = np.abs(im) > th_contrast
        mask_r = PrincipalCurvature[level] < th_r
        mask_unified = mask_c & mask_r

        valid_coords = np.where(mask_unified == True)
        for pt in zip(valid_coords[0], valid_coords[1]):
            currLevelVal = im[pt]
            if isSpatialExtrema(pt, level, DoGPyramid) is False:
                continue
            currLevelNeighbors = findEightNeighbors(pt)
            if isLevelExtrema(currLevelVal, currLevelNeighbors, im, max_x, max_y) is True:
                locsDoG.append(pt + (level,))

    return locsDoG


###############################################################################
def isSpatialExtrema(pt, level, DoGPyramid):
###############################################################################
    maxLevel = len(DoGPyramid) - 1
    currVal = DoGPyramid[level][pt]

    if level == maxLevel:
        prevVal = DoGPyramid[level - 1][pt]
        if currVal > prevVal or currVal < prevVal:
            return True
        else:
            return False
    elif level == 0:
        nextVal = DoGPyramid[level + 1][pt]
        if currVal > nextVal or currVal < nextVal:
            return True
        else:
            return False
    else:
        nextVal = DoGPyramid[level + 1][pt]
        prevVal = DoGPyramid[level - 1][pt]
        if (currVal > prevVal and currVal > nextVal) or \
            (currVal < prevVal and currVal < nextVal):
            return True
        else:
            return False


###############################################################################
def findEightNeighbors(pt):
###############################################################################
    x, y = pt
    pt1 = (x - 1, y - 1) 
    pt2 = (x, y - 1)
    pt3 = (x + 1, y - 1)
    pt4 = (x - 1, y)
    pt5 = (x + 1, y)
    pt6 = (x - 1, y + 1)
    pt7 = (x, y + 1)
    pt8 = (x + 1, y + 1)
    return pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8
    

###############################################################################
def isLevelExtrema(currLevelVal, currLevelNeighbors, im, max_x, max_y):
###############################################################################
    vals = []
    for p in currLevelNeighbors:
        x, y = p
        if x > max_x or x < 0 or y > max_y or y < 0:
            continue
        vals.append(im[p])

    max_val = max(vals)
    min_val = min(vals)
    if currLevelVal > max_val or currLevelVal < min_val:
        return True
    else:
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