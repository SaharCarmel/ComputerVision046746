###############################################################################
#                            pkg imports                                      #
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scipyio
import cv2
import os
from my_keypoint_det import *
from scipy.spatial.distance import cdist


###############################################################################
#                            functions                                        #
###############################################################################
def makeTestPattern(patchWidth, nbits):
    # Computes pairs of X,Y coordinates using uniform distribution
    # 
    # INPUTS
    # patchWidth - The width of the patch we randomize coordinates from
    # nbits - The descriptor length 
    #
    # OUPUTS
    # compareX, compareY - Uniformly distributed coordinates of X and Y
    compareX = np.random.randint(
        low     = (-patchWidth/2), 
        high    = (patchWidth/2), 
        size    = (2, nbits)
    )
    compareY = np.random.randint(
        low     = (-patchWidth/2), 
        high    = (patchWidth/2), 
        size    = (2, nbits)
    )
    return compareX, compareY


def computeBrief(im, GaussianPyramid, locsDoG, k, 
    levels, compareX, compareY):
    # Computer brief descriptor of an image
    # 
    # INPUTS
    # im {np.array} -- grayscale image
    # GaussianPyramid {im list} -- DoG pyramid of image
    # locsDoG {features location (level, x , y)} -- position of features
    # k {float} -- hyperparam
    # levels {list} -- levels of DoG
    # compareX {list, shape:[2,nbits]} -- list of pixel to compare
    # compareY {, shape[2,bits]} -- 2nd list of pixel to compare
    # 
    # OUPUTS
    # locs  -- locs of legitimate locations
    # desc  -- a M x N matrix of nbits string
    locs = []
    desc = []
    maxX = np.max(compareX)
    minX = np.min(compareX)
    maxY = np.max(compareY)
    minY = np.min(compareY)
    bigX = np.max([maxX, np.abs(minX)])
    bigY = np.max([maxY, np.abs(minY)])
    for loc in locsDoG:
        if locInBoundry(im.shape, loc, bigX, bigY):
            locs.append([loc[2], loc[1], loc[0]])
            desc.append(calculateBRIEF(GaussianPyramid, 
                    loc, 
                    compareX, 
                    compareY
                )
            )
    locs = np.array(locs)
    desc = np.array(desc)
    return locs, desc


def locInBoundry(imsize, loc, bigX, bigY):
    # Check if a pixel is inside the image boundaries
    # 
    # INPUTS
    # imsize - Size of the image (height and width)
    # loc - loc containing both X and Y coordinates
    # bigX, bigY - boundaries to the image in both X and Y direction
    #  
    #
    # OUPUTS
    # Boolean value - True if pixel is in boundaries or False if its out
    #                 of the picture boundaries
    x = loc[1]
    y = loc[2]
    if (x + bigX < imsize[0] and x - bigX > 0) and \
        (y + bigY < imsize[1] and y - bigY > 0):
        return True
    else:
        return False


def calculateBRIEF(GaussianPyramid, loc, compareX, compareY):
    descriptor = []
    for i in range(compareX.shape[1]):
        x = compareX[:, i]
        y = compareY[:, i]
        if GaussianPyramid[loc[0]][loc[1] + x[0],loc[2] + x[1]] < \
            GaussianPyramid[loc[0]][loc[1] + y[0],loc[2]+ y[1]]:
            descriptor.append(1)
        else:
            descriptor.append(0)
    return descriptor


def briefLite(im):
    # A wrapper which takes an image, converts it to gray, creates locsDoG
    # and a DoG pyramid and then calls computeBrief using those variables
    # 
    # INPUTS
    # im - Image we want to find it's BRIEF descriptor
    #
    # OUPUTS
    # see computeBrief(...)
    fp_module = os.path.dirname(__file__)
    testpatternPath = os.path.join(fp_module, 'testpattern.mat')      
    testPattern = scipyio.loadmat(testpatternPath)

    grayImage = convertToGrayNormalize(im)
    locsDoG, DoGPyramid = DoGdetector(
        im          = grayImage, 
        sigma0      = sigma0,
        k           = k,
        levels      = levels,
        th_contrast = thetaC,
        th_r        = thetaR
    )
    locs, desc = computeBrief(
        im              = grayImage, 
        GaussianPyramid = DoGPyramid, 
        locsDoG         = locsDoG,
        k               = k,
        levels          = levels,
        compareX        = testPattern['compareX'], 
        compareY        = testPattern['compareY']
    )
    
    return locs, desc