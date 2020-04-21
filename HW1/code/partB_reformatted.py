###############################################################################
#                            pkg imports                                      #
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scipyio
import cv2
import os
from partA_reformatted import *
from scipy.spatial.distance import cdist


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
def makeTestPattern(patchWidth, nbits):
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


def computeBrief(im, GaussianPyramid, locsDoG, k, levels, compareX, compareY):
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
    x = loc[1]
    y = loc[2]
    if (x + bigX < imsize[0] and x - bigX > 0) and (y + bigY < imsize[1] and y - bigY > 0):
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
    #hyperparams = scipyio.loadmat('code/hyperparams.mat')
    testpatternPath = os.path.join(os.getcwd(), 'testpattern.mat')
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


def briefMatch(desc1, desc2, ratio):
    D = cdist(
        np.float32(desc1), 
        np.float32(desc2), 
        metric = 'hamming'
    )
    # find smallest distance
    ix2 = np.argmin(D, axis = 1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis = 1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2 + 1e-10)
    is_discr = r < ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]
    matches = np.stack((ix1, ix2), axis = -1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure(figsize=(15,15))
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1] + im2.shape[1]), dtype = 'uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap = 'gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r', lw = 0.1)
        plt.plot(x,y,'g.', lw = 0.1)
    plt.show()


###############################################################################
#                       bonus section functions                               #
###############################################################################
def makeRotationIm(img, startAngle = 0, stopAngle = 90, n = 10):
    angles = np.linspace(startAngle, stopAngle, n)
    rows, cols, _ = img.shape
    imgs = []
    for angle in angles:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        imgs.append([dst, angle])
    return imgs

def testRot(im, startAngle = 0, stopAngle = 90, n = 10):
    matches = []
    angles = []
    rotImages = makeRotationIm(im, startAngle, stopAngle, n)
    baseloc, basedesc = briefLite(im)
    for img, angle in rotImages:
        locs, desc = briefLite(img)
        _match = briefMatch(basedesc, desc, 0.8)
        matches.append(len(_match))
        angles.append(angle)
    return matches, angles


###############################################################################
#                                main                                         #
###############################################################################
def main():
    fp_im1 = os.path.join(os.getcwd(),'..', 'data','pf_desk.jpg')
    fp_im2 = os.path.join(os.getcwd(),'..', 'data','pf_scan_scaled.jpg')
    im1 = loadImage(fp_im1)
    im2 = loadImage(fp_im2)

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2, ratio = 0.9)

    plotMatches(im1, im2, matches, locs1, locs2)


###############################################################################
#                               bonus                 
###############################################################################
def bonus():
    im1 = cv2.imread('data/chickenbroth_01.jpg')
    matches, angles = testRot(im1)
    plt.figure(figsize = (15,15))
    plt.bar(angles, matches)
    plt.title('# of matches vs rotation angle')
    plt.xlabel('angle')
    plt.ylabel('# of matches')

if __name__ == "__main__":
    # main()
    bonus()