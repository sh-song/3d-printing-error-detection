import cv2
import numpy as np
from skimage import filters, data, segmentation, color
rect_top_img = cv2.imread('images/real.jpeg', 0)

print('==============================',rect_top_img.shape)


# Apply median filtration to reduce amount of visual noise
filtered_top_view = filters.median(rect_top_img,selem=np.ones((9,9)))


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # print('==========median', v)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

edges_new = auto_canny(filtered_top_view)#*(np.flip(stl_mask,0))

print(edges_new.shape)

# cv2.imshow('edges.png', edges_new)
# cv2.waitKey(0)