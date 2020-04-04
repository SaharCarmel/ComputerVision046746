#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2

# %matplotlib inline

im = cv2.imread('../data/model_chickenbroth.jpg')
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
_ = plt.axis('off')
plt.show()

#%%

def createGaussianPyramid(im, sigma0, k, levels):
    GaussianPyramid = []
    for i in range(len(levels)):
        sigma_ = sigma0 * k ** levels[i]
        size = int(np.floor( 3 * sigma_ * 2) + 1)
        blur = cv2.GaussianBlur(im,(size,size),sigma_)
        GaussianPyramid.append(blur)
    return np.stack(GaussianPyramid)

def displayPyramid(pyramid):
    plt.figure(figsize=(16,5))
    plt.imshow(np.hstack(pyramid), cmap='gray')
    plt.axis('off')

def plotCv2Image(im):
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    _ = plt.axis('off')
    plt.show()

#%%
sigma0 = 1
k = np.sqrt(2)
levels = [-1, 0, 1, 2, 3, 4]
norm_image = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
grayscale_im = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
# grayscale_norm_im = grayscale_im / np.max(grayscale_im)
plotCv2Image(grayscale_im)
pyramid = createGaussianPyramid(grayscale_im, sigma0=sigma0, k=k, levels=levels)
displayPyramid(pyramid)
filename = 'sec1.1'


# %%
