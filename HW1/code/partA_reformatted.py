###############################################################################
#                            pkg imports                                      #
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


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
def loadImage(fp):
    im = cv2.imread(fp)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # _ = plt.axis('off')
    # plt.show()
    # im = cv2.GaussianBlur(im, (11, 11), 1)
    return im


def createGaussianPyramid(im, sigma0, k, levels):
    GaussianPyramid = []

    for i in range(len(levels)):
        sigma_ = sigma0 * k ** levels[i]
        size = int(np.floor(3 * sigma_ * 2) + 1)
        blur = cv2.GaussianBlur(im, (size, size), sigma_)
        GaussianPyramid.append(blur)

    return np.stack(GaussianPyramid)



def displayPyramid(pyramid):
    plt.figure(figsize = (16, 5))
    plt.imshow(np.hstack(pyramid), cmap = 'gray')
    plt.axis('off')
    plt.show()


def convertToGrayNormalize(im):
    GRAY_MAX_VAL = 255
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im / GRAY_MAX_VAL
    return im


def createDoGPyramid(GaussianPyramid, levels):
    DoGPyramid = []

    for i, x in enumerate(GaussianPyramid):
        if i == 0:
            continue
        DoGPyramid.append(GaussianPyramid[i] - GaussianPyramid[i - 1])
    DoGLevels = levels

    return DoGPyramid, DoGLevels


def computePrincipalCurvature(DoGPyramid):
    PrincipalCurvature = []

    for _filter in DoGPyramid:
        _tempMat = np.zeros_like(_filter)
        Dxx, Dyy, Dxy = getSecondDerivative(_filter)
        for x, y in np.ndindex(Dxx.shape):
            Dxx_i = Dxx[x, y]
            Dyy_i = Dyy[x, y]
            Dxy_i = Dxy[x, y]
            _tempMat[x, y] = calculateCurvatureRatio(Dxx_i, Dyy_i, Dxy_i)
        PrincipalCurvature.append(_tempMat)

    return PrincipalCurvature


def getSecondDerivative(im):
    K_SIZE = 3
    Dxx = cv2.Sobel(im, cv2.CV_64F, dx = 2, dy = 0, ksize = K_SIZE)
    Dyy = cv2.Sobel(im, cv2.CV_64F, dx = 0, dy = 2, ksize = K_SIZE)
    Dxy = cv2.Sobel(im, cv2.CV_64F, dx = 1, dy = 1, ksize = K_SIZE)
    return Dxx, Dyy, Dxy


def calculateCurvatureRatio(Dxx, Dyy, Dxy):
    tr = Dxx + Dyy
    det = Dxx * Dyy - Dxy * Dxy
    if det == 0:
        det = np.finfo(float).eps
    R = tr ** 2 / det
    return R


def getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature, 
    th_contrast, th_r):
    locsDoG = []

    max_x, max_y = DoGPyramid[0].shape
    max_x-=1
    max_y-=1

    for level, im in enumerate(DoGPyramid):
        mask_c = np.abs(im) > th_contrast
        mask_r = np.abs(PrincipalCurvature[level]) < th_r
        mask_unified = mask_c & mask_r

        valid_pixels = np.where(mask_unified == True)
        for pt in zip(valid_pixels[0], valid_pixels[1]):
            currLevelVal = im[pt]
            spatialNeighborsVals = getSpatialVals(pt, level, DoGPyramid, max_x, max_y)
            max_spatial = max(spatialNeighborsVals)
            min_spatial = min(spatialNeighborsVals)
            if currLevelVal < min_spatial or currLevelVal > max_spatial:
                locsDoG.append((level,) + pt)

    return locsDoG


def getSpatialVals(pt, level, DoGPyramid, max_x, max_y):
    maxLevel = len(DoGPyramid) - 1
    vals = []

    if level != maxLevel:
        vals.append(DoGPyramid[level + 1][pt])
    if level != 0:
        vals.append(DoGPyramid[level - 1][pt])

    sameLevelNeighbors = findEightNeighbors(pt)
    for p in sameLevelNeighbors:
        x, y = p
        if x > max_x or x < 0 or y > max_y or y < 0:
            continue
        vals.append(DoGPyramid[level][p])

    return vals


def findEightNeighbors(pt):
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
    

def DoGdetector(im, sigma0, k, levels, th_contrast, th_r):
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
#                                main                                         #
###############################################################################
def main():
    fp_im = os.path.join(os.getcwd(), 'data','polygons.jpg')
    im = loadImage(fp_im)
    im_grayscale = convertToGrayNormalize(im)
    locsDoG, GaussianPyramid = DoGdetector(
        im_grayscale, 
        sigma0, 
        k, 
        levels, 
        thetaC, 
        thetaR
    )

    plt.figure(figsize=(15,15))
    plt.imshow(im)
    x = []
    y = []
    for pt in locsDoG:
        x.append(pt[1])
        y.append(pt[0])
    plt.scatter(x, y, marker = 'o', c = 'g')
    plt.show()

if __name__ == "__main__":
    main()

# %%
