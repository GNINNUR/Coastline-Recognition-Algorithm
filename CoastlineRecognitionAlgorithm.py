import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
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
originalImg = cv2.imread(options.image, 1)

## Color Filter
# Min/Max For Color Filter
BLUE_MIN = np.array([80, 110, 0],np.uint8)
BLUE_MAX = np.array([160, 255, 255],np.uint8)

#Convert from RGB to HSV
hsv_img = cv2.cvtColor(originalImg, cv2.COLOR_BGR2HSV)

#Apply Color Filter
colorFilterImg = cv2.inRange(hsv_img, BLUE_MIN, BLUE_MAX)

## Gaussian Blur 
gaussianBlurImage = cv2.GaussianBlur(originalImg,(5,5),0)

## Gaussian Blur With Color Filter
gaussianBlurColorFilterImg = cv2.GaussianBlur(colorFilterImg,(5,5),0)
         
## Canny Edge Detector With Color Filter
cannyEdgeColorFilterImg = feature.canny(colorFilterImg)

## Canny Edge Detector With Gaussian Blur And Color Filter
#arrayGaussianBlurColorFilterImg = np.array(gaussianBlurColorFilterImg,0)
arrayGaussianBlurColorFilterImg = np.asarray(gaussianBlurColorFilterImg)
cannyEdgeGaussianBlurColorFilterImg = feature.canny(arrayGaussianBlurColorFilterImg)

## Bilateral Filter
bilateralFilterImg = cv2.bilateralFilter(originalImg,9,75,75)

## Bilateral Filter With Color Filter
bilateralFilterColorFilterImg = cv2.bilateralFilter(colorFilterImg,9,75,75)

## Canny Edge With Bilateral And Color Filter
cannyEdgeBilateralFilterColorFilterImg = feature.canny(bilateralFilterColorFilterImg)
## Display results
fig, (img0, img1, img2, img6) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3))
fig, (img3, img4, img5, img7) = plt.subplots(nrows=1, ncols=4, figsize=(8, 3))

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
img7.set_title('Canny Edge + Bilateral Filter + Color Filter' ,fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()


