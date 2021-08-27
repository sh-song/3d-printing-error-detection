from skimage import filters
import cv2
import numpy as np

class EdgeDetector:
    def __init__(self):
        pass
    
    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # print('==========median', v)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        # return the edged image
        return edged

    def run(self, image, vis=True):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print('=================shape', gray_image.shape)

        filtered_image = filters.median(gray_image,selem=np.ones((9,9)))


        edged_image = self.auto_canny(filtered_image)#*(np.flip(stl_mask,0))


        if vis:
            cv2.imshow('edged_image', edged_image)
            cv2.waitKey(0)
        
        return edged_image

