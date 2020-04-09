# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %matplotlib inline

im = cv2.imread('../data/model_chickenbroth.jpg')
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')
plt.show()

# %%


def createGaussianPyramid(im, sigma0, k, levels):
    GaussianPyramid = []
    for i in range(len(levels)):
        sigma_ = sigma0 * k ** levels[i]
        size = int(np.floor(3 * sigma_ * 2) + 1)
        blur = cv2.GaussianBlur(im, (size, size), sigma_)
        GaussianPyramid.append(blur)
    return np.stack(GaussianPyramid)


def displayPyramid(pyramid):
    plt.figure(figsize=(16, 5))
    plt.imshow(np.hstack(pyramid), cmap='gray')
    plt.axis('off')


def plotCv2Image(im, title='', saveFlag=False, savePath=''):
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    _ = plt.axis('off')
    plt.title(title)
    plt.show()
    if saveFlag:
        plt.savefig(savePath+title+'.png')


# %% 1.1 section
sigma0 = 1
k = np.sqrt(2)
levels = [-1, 0, 1, 2, 3, 4]
norm_image = cv2.normalize(im, None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
grayscale_im = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
# grayscale_norm_im = grayscale_im / np.max(grayscale_im)
# plotCv2Image(grayscale_im)
pyramid = createGaussianPyramid(
    grayscale_im, sigma0=sigma0, k=k, levels=levels)
displayPyramid(pyramid)
filename = 'sec1.1.png'


# %%
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
        if i == len(GaussianPyramid)-1:
            break
        DoGPyramid.append(GaussianPyramid[i] - GaussianPyramid[i+1])
    DoGLevels = levels
    return DoGPyramid, DoGLevels


# %% sec 1.3
DoGpyr, _ = createDoGPyramid(pyramid, levels=levels)
displayPyramid(DoGpyr)
plt.savefig('../report/media/'+'sec1-3.png')

# %%


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
        sobelxx, sobelyy, sobelxy = getSecondDev(_filter)
        _tempMat = np.zeros_like(_filter)
        for i in range(_filter.shape[0]):
            for j in range(_filter.shape[1]):
                _tempMat[i, j] = calculateCurvature(
                    sobelxx[i, j], sobelyy[i, j], sobelxy[i, j])
        PrincipalCurvature.append(_tempMat)
    return PrincipalCurvature


def getSecondDev(im):
    sobelxx = cv2.Sobel(im, cv2.CV_64F, 2, 0, ksize=3)
    sobelyy = cv2.Sobel(im, cv2.CV_64F, 0, 2, ksize=3)
    sobelxy = cv2.Sobel(im, cv2.CV_64F, 1, 1, ksize=3)
    return sobelxx, sobelyy, sobelxy


def assembleHessianTraceDet(xx, yy, xy):
    hessian = np.array([[xx, xy], [xy, yy]])
    trace = hessian.trace()
    det = np.linalg.det(hessian)
    return hessian, trace, det


def calculateCurvature(sobelxx, sobelyy, sobelxy):
    hessian, trace, det = assembleHessianTraceDet(sobelxx, sobelyy, sobelxy)
    curvature = np.power(trace, 2) / det
    return curvature

# %%
# sobely = cv2.Sobel(grayscale_im,cv2.CV_64F,0,2,ksize=3)
# plt.imshow(sobely,cmap = 'gray')
# sobelx = cv2.Sobel(grayscale_im,cv2.CV_64F,2,0,ksize=5)
# plt.imshow(sobelx,cmap = 'gray')
# sobelxy = cv2.Sobel(grayscale_im,cv2.CV_64F,1,1,ksize=5)
# plt.imshow(sobelxy,cmap = 'gray')


# %%
curavaturePyr = computePrincipalCurvature(DoGPyramid=DoGpyr)
displayPyramid((curavaturePyr))

# %%


def getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature,
                    th_contrast, th_r):
    # Returns local extrema points in both scale and space using the DoGPyramid
    # INPUTS
    # DoG_pyramid - size (len(levels) - 1, imH, imW ) matrix of the DoG pyramid
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
    for _dog_filter, _pc_filter in zip(DoGPyramid, PrincipalCurvature):
        theta_c_term = np.abs(_dog_filter)> th_contrast
        theta_r_term = _pc_filter < th_r
        total_term = theta_c_term & theta_r_term
    return locsDoG
