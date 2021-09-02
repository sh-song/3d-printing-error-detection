import numpy as np
import numpy.linalg as la

#--- for data visualization
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits import mplot3d

#--- for image processing
import cv2
import scipy
from scipy import ndimage
import skimage
from skimage import filters, data, segmentation, color
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.future import graph

#--- for stl processing
import meshcut
import stl
from stl import mesh

#--- for gcode parser
from pygcode import *
from pygcode import Line
from pygcode import Machine, GCodeRapidMove

#--- for texture segmentation
from sklearn.mixture import GaussianMixture

#--- for multitemplate matching
import MTM
from MTM import matchTemplates, drawBoxesOnRGB

#--- for agglomerative clustering
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

#--- for convex hull of failures
from scipy.spatial import ConvexHull, convex_hull_plot_2d
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

# Create mask from STL
# Apply additional transformation to the contour if necessary

def SRT(point_cloud,scale,theta,tx,ty):
    S = scale
    R = np.array([[np.cos(theta),np.sin(theta)],\
                  [-np.sin(theta),np.cos(theta)]])
    T = np.array([tx,ty])
    transformed = S*R.dot(point_cloud)+T[:,np.newaxis]
    return transformed


# Generate a contour template for Multi Template Matching

# here we create two masks (outer and inner) from the gcode type:wall-inner
# then substact "inner" from outer" to get the gcode-precise outline




X_active_wall_inner = [109.618, 102.716, 102.716, 109.618, 109.218, 103.116, 103.116, 109.218, 92.501
, 88.752, 99.26, 116.116, 126.626, 122.876, 107.688, 92.154, 88.319, 99.067
, 116.309, 127.059, 123.223, 107.688, 90.204, 108.659, 126.033, 106.442, 125.335
, 98.806, 91.876, 102.635, 94.314, 107.613, 117.805, 109.697, 122.88, 109.018
, 94.988, 114.316, 124.981, 115.041, 103.361, 91.804, 111.616, 99.379, 89.032]

Y_active_wall_inner = [87.86, 87.86, 94.762, 94.762, 88.26, 88.26, 94.362, 94.362, 103.422
, 86.989, 73.81, 73.81, 86.989, 103.422, 110.736, 103.699, 86.89, 73.41
, 73.41, 86.89, 103.699, 111.18, 92.998, 73.889, 89.229, 110.047, 85.498
, 106.37, 100.327, 91.228, 80.138, 87.78, 76.057, 89.822, 103.047, 94.842
, 104.531, 73.889, 93.835, 107.106, 94.842, 83.285, 108.754, 73.889, 87.86]
 
 
 
#  # x
# [109.218 103.116 103.116 109.218  92.154  88.319  99.067 116.309 127.059
#  123.223 107.688]
# # y 
# [ 88.26   88.26   94.362  94.362 103.699  86.89   73.41   73.41   86.89
#  103.699 111.18 ]

print(X_active_wall_inner, Y_active_wall_inner)
rect_top_img = cv2.imread('images/filtered_top_view.png')


gcode_wall_inner_shape = np.vstack([(X_active_wall_inner),(Y_active_wall_inner)])
gcode_wall_inner_shape = SRT(gcode_wall_inner_shape, 5.7,0.2,273,231)
gcode_wall_inner_mask_points = np.asarray(gcode_wall_inner_shape.T.reshape((-1,2)),dtype=np.int32)



gcode_mask_inner = np.zeros((rect_top_img.shape), np.uint8)
cv2.polylines(gcode_mask_inner,[gcode_wall_inner_mask_points],False,(200,200,200),4)

gcode_mask_outer = np.zeros((rect_top_img.shape), np.uint8)
cv2.polylines(gcode_mask_outer,[gcode_wall_inner_mask_points],False,(200,200,200),6)


print(np.shape(rect_top_img))
# print(np.shape(gcode_mask_outer))
print(np.shape(gcode_mask_inner))

final_mask = np.flip(gcode_mask_outer,0)*np.flip(gcode_mask_inner,0)

# Find the part using MTM

template_to_match = np.flip(gcode_mask_outer,0)-np.flip(gcode_mask_inner,0)
template_to_match = template_to_match[120:390,142:412]
print(template_to_match.shape)

filtered_top_view = cv2.imread('images/filtered_top_view.png')
print(np.max(template_to_match))
print(np.max(auto_canny(filtered_top_view)))

# Format the template into a list of tuple (label, templateImage)
LAYER_NUMBER = 5
listTemplate = [("L"+str(LAYER_NUMBER),template_to_match)]

# Find a match template in the image, if any
Hits = matchTemplates(listTemplate,auto_canny(filtered_top_view),score_threshold=0.01,\
                      method=cv2.TM_CCOEFF_NORMED,maxOverlap=0)

print("Found {} hits".format( len(Hits.index) ) )
print(Hits)

Overlay = drawBoxesOnRGB(auto_canny(filtered_top_view),Hits,showLabel=False)

Overlay[Hits.BBox[0][1]:Hits.BBox[0][1]+Hits.BBox[0][2],\
        Hits.BBox[0][0]:Hits.BBox[0][0]+Hits.BBox[0][3]] = (80,100,100)