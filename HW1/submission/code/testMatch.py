###############################################################################
#                            pkg imports                                      #
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scipyio
import cv2
import os
from my_keypoint_det import loadImage
from my_BRIEF import *
from scipy.spatial.distance import cdist


###############################################################################
#                            functions                                        #
###############################################################################
def briefMatch(desc1, desc2, ratio):
    # performs the descriptor matching
    # inputs : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in ima
    # ge 1 and 2.
    # n is the number of bits in the brief
    # outputs : matches - p x 2 matrix. where the first column are indices
    # into desc1 and the second column are indices into desc2
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
#                                main                                         #
###############################################################################
def main():
    fp_module = os.path.dirname(__file__)
    fp_im1 = os.path.join(fp_module, 'data', 'model_chickenbroth.jpg')
    fp_im2 = os.path.join(fp_module, 'data', 'model_chickenbroth.jpg')
    im1 = loadImage(fp_im1)
    im2 = loadImage(fp_im2)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2, ratio = 0.9)
    plotMatches(im1, im2, matches, locs1, locs2)


if __name__ == "__main__":
    main()