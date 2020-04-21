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
    # Takes in a patch to an image, loads it and the converts it from BGR
    # to RGB
    #
    # INPUTS
    # fp - path to image
    #
    # OUTPUTS
    # im - RGB image object
    im = cv2.imread(fp)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
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
    # Takes in an image, converts it to gray and the normalizes the pixel
    # values to [0, 1] by dividing each pixel by 255
    # INPUTS
    # im - image
    #
    # OUTPUTS
    # gray_im - gray and normalized picture
    GRAY_MAX_VAL = 255
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray_im = gray_im / GRAY_MAX_VAL
    return gray_im


def createDoGPyramid(GaussianPyramid, levels):
    # Produces DoG Pyramid
    # inputs
    # Gaussian Pyramid - A matrix of grayscale images of size
    #                    (len(levels), shape(im))
    # levels           - the levels of the pyramid where the blur at each level is
    #                    outputs
    # DoG Pyramid      - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    #                    created by differencing the Gaussian Pyramid input
    DoGPyramid = []

    for i, x in enumerate(GaussianPyramid):
        if i == 0:
            continue
        DoGPyramid.append(GaussianPyramid[i] - GaussianPyramid[i - 1])
    DoGLevels = levels

    return DoGPyramid, DoGLevels


def computePrincipalCurvature(DoGPyramid):
    # Edge Suppression
    # Takes in DoGPyramid generated in createDoGPyramid and returns
    # PrincipalCurvature,a matrix of the same size where each point contains the
    # curvature ratio R for the corre-sponding point in the DoG pyramid
    #
    # INPUTS
    # DoG Pyramid - size (len(levels) - 1, shape(im)) matrix of the DoG pyramid
    #
    # OUTPUTS
    # PrincipalCurvature - size (len(levels) - 1, shape(im)) matrix where each
    # point contains the curvature ratio R for the
    # corresponding point in the DoG pyramid
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
    # Takes in an image and returns three matrix each containing a second
    # degree derivative using Sobel algorithm with a kernel size of 3
    #
    # INPUTS
    # im - image object
    #
    # OUTPUTS
    # Dxx - Matrix where each element represents the second derivative 
    #       in the X direction
    # Dyy - Matrix where each element represents the second derivative 
    #       in the Y direction
    # Dyy - Matrix where each element represents the the derivative in X
    #       and Y direction
    K_SIZE = 3
    Dxx = cv2.Sobel(im, cv2.CV_64F, dx = 2, dy = 0, ksize = K_SIZE)
    Dyy = cv2.Sobel(im, cv2.CV_64F, dx = 0, dy = 2, ksize = K_SIZE)
    Dxy = cv2.Sobel(im, cv2.CV_64F, dx = 1, dy = 1, ksize = K_SIZE)
    return Dxx, Dyy, Dxy


def calculateCurvatureRatio(Dxx, Dyy, Dxy):
    # Takes in three matrix which represent the image derivatives in given
    # directions and calculates the curvature ratio pixelwise.
    # In case we get a determinant of zero we simply replace it with epsilon
    # to avoid division by zero
    #
    # INPUTS
    # Dxx, Dyy, Dxy - See getSecondDerivative(im)
    #
    # OUTPUTS
    # R - Curvature ratio for a given pixel
    tr = Dxx + Dyy
    det = Dxx * Dyy - Dxy * Dxy
    if det == 0:
        det = np.finfo(float).eps
    R = tr ** 2 / det
    return R


def getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature, 
    th_contrast, th_r):
    # Returns local extrema points in both scale and space using the DoGPyramid
    # INPUTS
    # DoGPyramid - size (len(levels) - 1, imH, imW ) matrix of the DoG pyramid
    # DoG_levels - The levels of the pyramid where the blur at each level is
    # outputs
    # principal_curvature - size (len(levels) - 1, imH, imW) matrix contains the
    # curvature ratio R
    # th_contrast - remove any point that is a local extremum but does not have a
    # DoG response magnitude above this threshold
    # th_r - remove any edge-like points that have too large a principal
    # curvature ratio
    # OUTPUTS
    # locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
    # scale and space, and also satisfies the two thresholds.
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
    # Retrieves the values of the 10-neighborhood pixels for a given pixel
    # If the current level is either the first or the last one, it skips
    # searching for value for the previous/next level 
    # For each neighbor in the same level we check the pixel is in bound
    # in case our pixel is at the edge of the image
    #
    # INPUTS
    # pt - (X, Y) coordinate which neighborhood we need
    # level - The level in the DoGPyramid of the given pt
    # DoGPyramid - size (len(levels) - 1, imH, imW ) matrix of the DoG pyramid
    # max_x, max_y - Last element in the X/Y direction, used to assert we're
    #                not trying to fetch values from outside the picture
    #
    # OUTPUTS
    # vals - A list containing all the values from the 10-neighborhood pixels
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
    # Retrieves the coordinates for all spatial neighbors of a given pixel
    #
    # INPUTS
    # pt - (X, Y) coordinate which neighborhood we need
    #
    # OUTPUTS
    # pt{1-8} - Coordinates of all 8 neighbors of pt in the same level
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
    # Putting it all together
    # Inputs Description
    # --------------------------------------------------------------------------
    # im Grayscale image with range [0,1].
    # sigma0 Scale of the 0th image pyramid.
    # k Pyramid Factor. Suggest sqrt(2).
    # levels Levels of pyramid to construct. Suggest -1:4.
    # th_contrast DoG contrast threshold. Suggest 0.03.
    # th_r Principal Ratio threshold. Suggest 12.
    # Outputs Description
    # --------------------------------------------------------------------------
    # locsDoG N x 3 matrix where the DoG pyramid achieves a local extrema
    # in both scale and space, and satisfies the two thresholds.
    # gauss_pyramid A matrix of grayscale images of size (len(levels),imH,imW)

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


def plotKeypoints(im, locsDoG, marker, color):
    plt.figure(figsize=(15,15))
    plt.imshow(im)
    x = []
    y = []
    for pt in locsDoG:
        x.append(pt[2])
        y.append(pt[1])
    plt.scatter(x, y, marker = marker, c = color)
    plt.show()


###############################################################################
#                                main                                         #
###############################################################################
def main():
    fp_module = os.path.dirname(__file__)
    fp_im = os.path.join(fp_module, 'data', 'model_chickenbroth.jpg')
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
    plotKeypoints(im, locsDoG, marker = 'o', color = 'g')


if __name__ == "__main__":
    main()