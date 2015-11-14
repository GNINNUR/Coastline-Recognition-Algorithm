import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from scipy import ndimage as ndi
import scipy
import cv2
from skimage import feature
from optparse import OptionParser



## Command-Line Parser
parser = OptionParser()
parser.add_option("-I", "--image", dest="image",
                  help="Image to be parsed", metavar="IMAGE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

(options, args) = parser.parse_args()

# Read Image
originalImg = cv2.imread(options.image)

## Color Filter
# Min/Max For Color Filter
BLUE_MIN = np.array([75, 60, 0],np.uint8)
BLUE_MAX = np.array([165, 255, 255],np.uint8)

#Convert from RGB to HSV
hsv_img = cv2.cvtColor(originalImg, cv2.COLOR_BGR2HSV)

#Apply Color Filter
colorFilterImg = cv2.inRange(hsv_img, BLUE_MIN, BLUE_MAX)

## Gaussian Blur 
gaussianBlurImage = cv2.GaussianBlur(originalImg,(5,5),0)

## Gaussian Blur With Color Filter
gaussianBlurColorFilterImg = cv2.GaussianBlur(colorFilterImg,(5,5),0)

## Image Denoising With Gaussian Blur And Color Filter
imageDenoisingGaussianBlurColorFilterImg = cv2.fastNlMeansDenoising(gaussianBlurColorFilterImg, h=200)
         
## Canny Edge Detector With Color Filter
cannyEdgeColorFilterImg = feature.canny(colorFilterImg)
cannyEdgeColorFilterImg2 = feature.canny(colorFilterImg, 2) # set sigma to 2

## Canny Edge Detector With Gaussian Blur And Color Filter
#arrayGaussianBlurColorFilterImg = np.array(gaussianBlurColorFilterImg,0)
arrayGaussianBlurColorFilterImg = np.asarray(gaussianBlurColorFilterImg)
cannyEdgeGaussianBlurColorFilterImg = feature.canny(arrayGaussianBlurColorFilterImg)
cannyEdgeGaussianBlurColorFilterImg2 = feature.canny(arrayGaussianBlurColorFilterImg, 2)

## Canny Edge Detector With Image Denoising And Gaussian Blur And Color Filter
arrayImageDenoisingGaussianBlurColorFilterImg = np.asarray(gaussianBlurColorFilterImg)
cannyEdgeImageDenoisingGaussianBlurColorFilterImg = feature.canny(arrayImageDenoisingGaussianBlurColorFilterImg)


## Bilateral Filter
bilateralFilterImg = cv2.bilateralFilter(originalImg,9,75,75)

## Bilateral Filter With Color Filter
bilateralFilterColorFilterImg = cv2.bilateralFilter(colorFilterImg,9,75,75)

## Image Denoising With Bilateral Filter And Color Filter
imageDenoisingBilateralFilterColorFilterImg = cv2.fastNlMeansDenoising(bilateralFilterColorFilterImg, h=200)

## Canny Edge With Bilateral And Color Filter
cannyEdgeBilateralFilterColorFilterImg = feature.canny(bilateralFilterColorFilterImg)
cannyEdgeGaussianBlurColorFilterImg2 = feature.canny(arrayGaussianBlurColorFilterImg, 2)

## Canny Edge With Image Denoising And Bilateral And Color Filter
arrayImageDenoisingBilateralFilterColorFilterImg = np.asarray(gaussianBlurColorFilterImg)
cannyEdgeImageDenoisingBilateralFilterColorFilterImg = feature.canny(arrayImageDenoisingBilateralFilterColorFilterImg)

## Write Desired Image to File

scipy.misc.imsave('cannyEdgeImageDenoisingBilateralFilterColorFilterImg.jpg', cannyEdgeImageDenoisingBilateralFilterColorFilterImg)


## Display results
fig, (img0, img1, img2, img6) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3))
fig, (img3, img8) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
fig, (img4, img5, img7) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
fig, (img9, img10) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
fig, (img11, img12) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
fig, (img13, img14) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

img0.imshow(originalImg, cmap=plt.cm.gray)
img0.axis('off')
img0.set_title('Original', fontsize=20)

img1.imshow(colorFilterImg, cmap=plt.cm.gray)
img1.axis('off')
img1.set_title('Colour Filter', fontsize=20)

img2.imshow(gaussianBlurImage, cmap=plt.cm.gray)
img2.axis('off')
img2.set_title('Gaussian Blur', fontsize=20)

img3.imshow(gaussianBlurColorFilterImg, cmap=plt.cm.gray)
img3.axis('off')
img3.set_title('Gaussian Blur + Color Filter', fontsize=20)

img4.imshow(cannyEdgeColorFilterImg, cmap=plt.cm.gray)
img4.axis('off')
img4.set_title('Canny Filter + Color Filter, $\sigma=1$', fontsize=20)

img5.imshow(cannyEdgeGaussianBlurColorFilterImg, cmap=plt.cm.gray)
img5.axis('off')
img5.set_title('Canny Edge + Gaussian Blur + Color Filter' ,fontsize=20)

img6.imshow(bilateralFilterImg, cmap=plt.cm.gray)
img6.axis('off')
img6.set_title('Bilateral Filter' ,fontsize=20)

img7.imshow(cannyEdgeBilateralFilterColorFilterImg, cmap=plt.cm.gray)
img7.axis('off')
img7.set_title('Canny Edge + Bilateral Filter + Color Filter' ,fontsize=20)\

img8.imshow(bilateralFilterColorFilterImg, cmap=plt.cm.gray)
img8.axis('off')
img8.set_title('Bilateral Filter + Color Filter' ,fontsize=20)

img9.imshow(cannyEdgeGaussianBlurColorFilterImg2, cmap=plt.cm.gray)
img9.axis('off')
img9.set_title('Canny Edge + Bilateral Filter + Color Filter, $\sigma=2$',fontsize=20)

img10.imshow(cannyEdgeGaussianBlurColorFilterImg2, cmap=plt.cm.gray)
img10.axis('off')
img10.set_title('Canny Edge + Bilateral Filter + Color Filter, $\sigma=2$' ,fontsize=20)

img11.imshow(imageDenoisingGaussianBlurColorFilterImg, cmap=plt.cm.gray)
img11.axis('off')
img11.set_title('Image Denoising + Gaussian Filter + Color Filter' ,fontsize=20)

img12.imshow(imageDenoisingBilateralFilterColorFilterImg, cmap=plt.cm.gray)
img12.axis('off')
img12.set_title('Image Denoising + Bilateral Filter + Color Filter' ,fontsize=20)

img13.imshow(cannyEdgeImageDenoisingBilateralFilterColorFilterImg, cmap=plt.cm.gray)
img13.axis('off')
img13.set_title('Canny Edge + Image Denoising + Bilateral Filter + Color Filter' ,fontsize=20)

img14.imshow(cannyEdgeImageDenoisingBilateralFilterColorFilterImg, cmap=plt.cm.gray)
img14.axis('off')
img14.set_title('Canny Edge + Image Denoising + Bilateral Filter + Color Filter' ,fontsize=20)


fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()


