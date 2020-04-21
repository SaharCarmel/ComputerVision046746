###############################################################################
#                            pkg imports                                      #
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from my_keypoint_det import loadImage
from testMatch import briefMatch
from my_BRIEF import *


###############################################################################
#                            functions                                        #
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
    fp_module = os.path.dirname(__file__)
    fp_im = os.path.join(fp_module, 'data', 'model_chickenbroth.jpg')    
    im = loadImage(fp_im)
    matches, angles = testRot(im)
    plt.figure(figsize = (15,15))
    plt.title('Number of matches as a function of the rotation angle',)
    plt.xlabel('Angle')
    plt.ylabel('Number of matches')
    plt.bar(angles, matches, width = 8)
    plt.show()


if __name__ == "__main__":
    main()
