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


# Intrinsic camera parameters (obtained on the calibration stage)
# Source images have already been undistorted

camera_intrinsic_K = np.array(
                         [[1552.3, 0,      650.1],
                         [0,       1564.8, 486.2],
                         [0,       0,      1]], dtype = "float")
print("Camera Intrinsic Parameters :\n {}".format(camera_intrinsic_K))



# Image count starts from Layer 2, where Layer 1 - is an empty layer
# G-Code starts with Layer 0
# Therefore, we use LAYER_NUMBER variable for image processing,
# and (LAYER_NUMBER-2) for G-Code processing

LAYER_NUMBER = 9 # Layer number displayed on captured image
LAYER_THICKNESS = 0.4 # mm

z_level = LAYER_THICKNESS*(LAYER_NUMBER-2)

if(LAYER_NUMBER<10):
    src = cv2.imread('dataset/rect_full/rect_L0%d.jpg' %LAYER_NUMBER, 0)
else:
    src = cv2.imread('dataset/rect_full/rect_L%d.jpg' %LAYER_NUMBER, 0)
    
img = src.copy()

print("Layer number displayed on captured image: LAYER_NUMBER = {}".format(LAYER_NUMBER))
print("Layer number saved in the G-Code file: LAYER_NUMBER-2 = {}".format(LAYER_NUMBER-2))
print("Height of the printed part: Z-level = LAYER_THICKNESS*(LAYER_NUMBER-2) = {:2f} mm".format(z_level))


# Image Projection: find cam transform based on 4 visual markers
# Manual calibration performed at the initial stage

#2D image points, [pixels]
image_points = np.array([
                            (391, 221),
                            (788, 77),
                            (1031, 368),
                            (569, 562),
                        ], dtype="double")
 
# 3D model points, [mm]
model_points = np.array([
                            (-44.0, -44.0, 0.0),
                            (-44.0, 44.0, 0.0),
                            (44.0, 44.0, 0.0),
                            (44.0, -44.0, 0.0)
                            ])

dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,\
                                            camera_intrinsic_K, dist_coeffs,flags=cv2.cv2.SOLVEPNP_ITERATIVE)

#print("Rotation Vector:\n {}".format(rotation_vector))
#print("\nTranslation Vector:\n {}".format(translation_vector))


# Extrinsic camera parameters

C = np.zeros((4,4), dtype=float)
C[0:3,0:3] = cv2.Rodrigues(rotation_vector)[0]
C[0:3,3] = translation_vector.T
C[3,3] = 1

print("Camera position relative to printing bed")
print("C = \n{}".format(np.round(C,3)))


# Plot figure to see how well the reference anchor points were specified

fig = plt.figure(figsize=(10,6), dpi=80)
plt.imshow(img, cmap='gray')

plt.scatter(391,221,c='springgreen',s=30)
plt.scatter(788,77,c='springgreen',s=30)
plt.scatter(1031,368,c='springgreen',s=30)
plt.scatter(569,562,c='springgreen',s=30)

plt.plot([391,788],[221,77],c='springgreen')
plt.plot([788,1031],[77,368],c='springgreen')
plt.plot([1031,569],[368,562],c='springgreen')
plt.plot([569,391],[562,221],c='springgreen')

plt.plot([391,1031],[221,368],c='springgreen',linestyle=':')
plt.plot([788,569],[77,562],c='springgreen',linestyle=':')
plt.title("Undistorted source frame | Image layer={} | G-Code layer={}".format(LAYER_NUMBER,LAYER_NUMBER-2))
plt.show()


# Compensate object rotation 
# (in case of the STL orientation in slicer does not match the G-Code orientation)

otheta_x = 0.0 # degrees
otheta_y = 0.0 # degrees
otheta_z = 169.0 # degrees

# X and Y shifts are functions of layer number, where the 
# coefficients were calibrated experimantally during the first setup
ot_x = LAYER_NUMBER * 3/172 - 617/215
ot_y = LAYER_NUMBER * (-1/325) + 42/13
ot_z = 0.0

oRx = np.array([[1,0,0],[0,np.cos(otheta_x*np.pi/180),-np.sin(otheta_x*np.pi/180)],\
               [0,np.sin(otheta_x*np.pi/180),np.cos(otheta_x*np.pi/180)]]) # rotation around x
oRy = np.array([[np.cos(otheta_y*np.pi/180),0,np.sin(otheta_y*np.pi/180)],[0,1,0],\
               [-np.sin(otheta_y*np.pi/180),0,np.cos(otheta_y*np.pi/180)]]) # rotation around x
oRz = np.array([[np.cos(otheta_z*np.pi/180),-np.sin(otheta_z*np.pi/180),0],\
               [np.sin(otheta_z*np.pi/180),np.cos(otheta_z*np.pi/180),0],[0,0,1]]) # rotation around x

oR = np.dot(np.dot(oRx,oRy),oRz)
ot = np.array([ot_x,ot_y,ot_z])

#print('oRx = \n{}\n'.format(oRx))
#print('oRy = \n{}\n'.format(oRy))
#print('oRz = \n{}\n'.format(oRz))
#print('\noR = \n{}\n'.format(oR))

#print('\not = \n{}\n'.format(ot.T))

H = np.zeros((4,4), dtype=float)
H[0:3,0:3] = oR
H[0:3,3] = ot.T
H[3,3] = 1

print("Printing object position relative to the build plate origin")
print('H = \n{}\n'.format(H))


# Various slicer programs provide different G-Code formats
# At this moment, the developed algorithm works only with G-Code generated by 
# "MatterControl" software (https://www.matterhackers.com/store/l/mattercontrol/sk/MKZGTDW6)

# Each G-Code category is assigned a specific number from 1 to 5:

# TYPE:WALL-OUTER          -- 1
# TYPE:WALL-INNER          -- 2
# TYPE:FILL                -- 3
# TYPE:SUPPORT             -- 4
# TYPE:SUPPORT-INTERFACE   -- 5


word_bank  = [] # e.g. "G1", "X15.03", etc.
layer_bank = [] # layer number: 0,1,2,...,N
type_bank  = [] # gcode type: 1,2,...,5.
line_bank  = [] # e.g. "G1 X0.01 Y0 Z0.4 F3000 ;comments"

parsed_Num_of_layers = 0
gcode_type = 0

with open('misc_data/70mm_low_poly_fox_MatterControl.gcode', 'r') as fh:
    for line_text in fh.readlines():
        line = Line(line_text) # all lines in file
        # print(line)
        w = line.block.words # splits blocks into XYZEF, omits comments
        # print(w)
        if(np.shape(w)[0] == 0): # if line is empty, i.e. comment line -> then skip it
            pass
        else:
            word_bank.append(w) # <Word: G01>, <Word: X15.03>, <Word: Y9.56>, <Word: Z0.269>, ...
            # print(word_bank) # does not process comments
            layer_bank.append(parsed_Num_of_layers)
            type_bank.append(gcode_type)
            line_bank.append(line_text)

        if line.comment:
            #print(line.comment)
            if (line.comment.text[0:6] == "LAYER:"):
                parsed_Num_of_layers = parsed_Num_of_layers + 1
                gcode_type = 0
            if line.comment:
                if (line.comment.text[0:15] == "TYPE:WALL-OUTER"):
                    #print("TYPE:WALL-OUTER")
                    gcode_type = 1
                if (line.comment.text[0:15] == "TYPE:WALL-INNER"):
                    #print("TYPE:WALL-INNER")
                    gcode_type = 2
                if (line.comment.text[0:9] == "TYPE:FILL"):
                    #print("TYPE:FILL")
                    gcode_type = 3
                if (line.comment.text[0:12] == "TYPE:SUPPORT"):
                    #print("TYPE:SUPPORT")
                    gcode_type = 4
                if (line.comment.text[0:22] == "TYPE:SUPPORT-INTERFACE"):
                    #print("TYPE:SUPPORT-INTERFACE")
                    gcode_type = 5
                
print("Number of parsed layers = {}".format(parsed_Num_of_layers))



# Project the G-Code to the source image to obtain the reference contour
# for vertical level validation and global contour matching

fig = plt.figure(figsize=(18,14), dpi=80)
plt.imshow(img,cmap='gray')

plt.scatter(391,221,c='springgreen',s=30)
plt.scatter(788,77,c='springgreen',s=30)
plt.scatter(1031,368,c='springgreen',s=30)
plt.scatter(569,562,c='springgreen',s=30)

plt.plot([391,788],[221,77],c='springgreen')
plt.plot([788,1031],[77,368],c='springgreen')
plt.plot([1031,569],[368,562],c='springgreen')
plt.plot([569,391],[562,221],c='springgreen')

plt.plot([391,1031],[221,368],c='springgreen',linestyle=':')
plt.plot([788,569],[77,562],c='springgreen',linestyle=':')


# for k in [6,8,70]: # for k in range(2,24,2) # draw for multiple layers
for k in [LAYER_NUMBER-1]: # layers
    command_bank = []
    line_command_bank = []
    gcode_line_number = 0
    
    # create additional arrays for gcode segmentation
    X_active_bank = []
    Y_active_bank = []
    Z_active_bank = []
    G_active_bank = []
    E_active_bank = []
    F_active_bank = []

    idx = []
    
    X_active_default = []; X_active_wall_outer = []; X_active_wall_inner = []
    Y_active_default = []; Y_active_wall_outer = []; Y_active_wall_inner = []
    Z_active_default = []; Z_active_wall_outer = []; Z_active_wall_inner = []
    G_active_default = []; G_active_wall_outer = []; G_active_wall_inner = []
    E_active_default = []; E_active_wall_outer = []; E_active_wall_inner = []
    F_active_default = []; F_active_wall_outer = []; F_active_wall_inner = []
    
    # auxiliary layer duplicates shifted by +1 and -1 layer height
    # for additional visual inspection that can be done by the user
    Z_active_wall_inner_top_aux = []; Z_active_wall_inner_bot_aux = []

    X_active_fill = []; X_active_support = []; X_active_support_interface = []
    Y_active_fill = []; Y_active_support = []; Y_active_support_interface = []
    Z_active_fill = []; Z_active_support = []; Z_active_support_interface = []
    G_active_fill = []; G_active_support = []; G_active_support_interface = []
    E_active_fill = []; E_active_support = []; E_active_support_interface = []
    F_active_fill = []; F_active_support = []; F_active_support_interface = []
    
    for i in range(len(layer_bank)): # for each line in file
        if (layer_bank[i] == k):
            idx.append(i)
            line_command_bank.append(line_bank[i])
            # line_command_bank = all gcode for the specific layer without comments
            for j in range(len(word_bank[i])):
                command_bank.append(str(word_bank[i][j]))
                if (str(word_bank[i][j])[:1] == 'G'):
                    G_active_bank.append(float(str(word_bank[i][j])[1:]))
                if (str(word_bank[i][j])[:1] == 'X'):
                    X_active_bank.append(float(str(word_bank[i][j])[1:]))
                if (str(word_bank[i][j])[:1] == 'Y'):
                    Y_active_bank.append(float(str(word_bank[i][j])[1:]))
                if (str(word_bank[i][j])[:1] == 'Z'):
                    Z_active_bank.append(float(str(word_bank[i][j])[1:]))
                if (str(word_bank[i][j])[:1] == 'E'):
                    E_active_bank.append(float(str(word_bank[i][j])[1:]))
                if (str(word_bank[i][j])[:1] == 'F'):
                    F_active_bank.append(float(str(word_bank[i][j])[1:]))

    for m in range(len(X_active_bank)):
        if(type_bank[np.min(idx)+m] == 0):
            X_active_default.append(X_active_bank[m])
            Y_active_default.append(Y_active_bank[m])
            Z_active_default.append(Z_active_bank[m])
        if(type_bank[np.min(idx)+m] == 1):
            X_active_wall_outer.append(X_active_bank[m])
            Y_active_wall_outer.append(Y_active_bank[m])
            Z_active_wall_outer.append(Z_active_bank[m])
        if(type_bank[np.min(idx)+m] == 2):
            X_active_wall_inner.append(X_active_bank[m])
            Y_active_wall_inner.append(Y_active_bank[m])
            Z_active_wall_inner.append(Z_active_bank[m])
            #--------------------
            Z_active_wall_inner_top_aux.append(Z_active_bank[m]+LAYER_THICKNESS*2)
            Z_active_wall_inner_bot_aux.append(Z_active_bank[m]-LAYER_THICKNESS*2)
            #--------------------
        if(type_bank[np.min(idx)+m] == 3):
            X_active_fill.append(X_active_bank[m])
            Y_active_fill.append(Y_active_bank[m])
            Z_active_fill.append(Z_active_bank[m])
        if(type_bank[np.min(idx)+m] == 4):
            X_active_support.append(X_active_bank[m])
            Y_active_support.append(Y_active_bank[m])
            Z_active_support.append(Z_active_bank[m])
        if(type_bank[np.min(idx)+m] == 5):
            X_active_support_interface.append(X_active_bank[m])
            Y_active_support_interface.append(Y_active_bank[m])
            Z_active_support_interface.append(Z_active_bank[m])

            
    G_default = np.zeros((np.shape(X_active_default)[0],4),dtype=np.float32)
    G_wall_outer = np.zeros((np.shape(X_active_wall_outer)[0],4),dtype=np.float32)
    G_wall_inner = np.zeros((np.shape(X_active_wall_inner)[0],4),dtype=np.float32)
    G_fill = np.zeros((np.shape(X_active_fill)[0],4),dtype=np.float32)
    G_support = np.zeros((np.shape(X_active_support)[0],4),dtype=np.float32)
    G_support_interface = np.zeros((np.shape(X_active_support_interface)[0],4),dtype=np.float32)
    
    #--------------------
    G_wall_inner_top_aux = np.zeros((np.shape(X_active_wall_inner)[0],4),dtype=np.float32)
    G_wall_inner_bot_aux = np.zeros((np.shape(X_active_wall_inner)[0],4),dtype=np.float32)
    #--------------------

    G_default[:,0] = X_active_default
    G_default[:,1] = Y_active_default
    G_default[:,2] = Z_active_default
    G_default[:,3] = np.ones((1,np.shape(X_active_default)[0]),dtype=np.float32)

    G_wall_outer[:,0] = X_active_wall_outer
    G_wall_outer[:,1] = Y_active_wall_outer
    G_wall_outer[:,2] = Z_active_wall_outer
    G_wall_outer[:,3] = np.ones((1,np.shape(X_active_wall_outer)[0]),dtype=np.float32)

    G_wall_inner[:,0] = X_active_wall_inner
    G_wall_inner[:,1] = Y_active_wall_inner
    G_wall_inner[:,2] = Z_active_wall_inner
    G_wall_inner[:,3] = np.ones((1,np.shape(X_active_wall_inner)[0]),dtype=np.float32)
    
    #-----------------------
    
    G_wall_inner_top_aux[:,0] = X_active_wall_inner
    G_wall_inner_top_aux[:,1] = Y_active_wall_inner
    G_wall_inner_top_aux[:,2] = Z_active_wall_inner_top_aux
    G_wall_inner_top_aux[:,3] = np.ones((1,np.shape(X_active_wall_inner)[0]),dtype=np.float32)
    
    G_wall_inner_bot_aux[:,0] = X_active_wall_inner
    G_wall_inner_bot_aux[:,1] = Y_active_wall_inner
    G_wall_inner_bot_aux[:,2] = Z_active_wall_inner_bot_aux
    G_wall_inner_bot_aux[:,3] = np.ones((1,np.shape(X_active_wall_inner)[0]),dtype=np.float32)
    
    #-----------------------

    G_fill[:,0] = X_active_fill
    G_fill[:,1] = Y_active_fill
    G_fill[:,2] = Z_active_fill
    G_fill[:,3] = np.ones((1,np.shape(X_active_fill)[0]),dtype=np.float32)

    G_support[:,0] = X_active_support
    G_support[:,1] = Y_active_support
    G_support[:,2] = Z_active_support
    G_support[:,3] = np.ones((1,np.shape(X_active_support)[0]),dtype=np.float32)

    G_support_interface[:,0] = X_active_support_interface
    G_support_interface[:,1] = Y_active_support_interface
    G_support_interface[:,2] = Z_active_support_interface
    G_support_interface[:,3] = np.ones((1,np.shape(X_active_support_interface)[0]),dtype=np.float32)
    
    tG_default = np.zeros((np.shape(G_default)[0],4), dtype=np.float32)
    tG_wall_outer = np.zeros((np.shape(G_wall_outer)[0],4), dtype=np.float32)
    tG_wall_inner = np.zeros((np.shape(G_wall_inner)[0],4), dtype=np.float32)
    tG_fill = np.zeros((np.shape(G_fill)[0],4), dtype=np.float32)
    tG_support = np.zeros((np.shape(G_support)[0],4), dtype=np.float32)
    tG_support_interface = np.zeros((np.shape(G_support_interface)[0],4), dtype=np.float32)
    #-----------------------
    tG_wall_inner_top_aux = np.zeros((np.shape(G_wall_inner_top_aux)[0],4), dtype=np.float32)
    tG_wall_inner_bot_aux = np.zeros((np.shape(G_wall_inner_bot_aux)[0],4), dtype=np.float32)
    
    for i in range(np.shape(G_default)[0]):
        tG_default[i] = np.dot(H,G_default[i])
    for i in range(np.shape(G_wall_outer)[0]):
        tG_wall_outer[i] = np.dot(H,G_wall_outer[i])
    for i in range(np.shape(G_wall_inner)[0]):
        tG_wall_inner[i] = np.dot(H,G_wall_inner[i])
    for i in range(np.shape(G_fill)[0]):
        tG_fill[i] = np.dot(H,G_fill[i])
    for i in range(np.shape(G_support)[0]):
        tG_support[i] = np.dot(H,G_support[i])
    for i in range(np.shape(G_support_interface)[0]):
        tG_support_interface[i] = np.dot(H,G_support_interface[i])
    #-----------------------
    for i in range(np.shape(G_wall_inner_top_aux)[0]):
        tG_wall_inner_top_aux[i] = np.dot(H,G_wall_inner_top_aux[i])
    for i in range(np.shape(G_wall_inner_bot_aux)[0]):
        tG_wall_inner_bot_aux[i] = np.dot(H,G_wall_inner_bot_aux[i])
        
        
    tGp_default = cv2.projectPoints(np.asarray(tG_default[:,0:3],dtype=float),rotation_vector,\
                        translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)
    tGp_wall_outer = cv2.projectPoints(np.asarray(tG_wall_outer[:,0:3],dtype=float),rotation_vector,\
                        translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)
    tGp_wall_inner = cv2.projectPoints(np.asarray(tG_wall_inner[:,0:3],dtype=float),rotation_vector,\
                        translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)
    tGp_fill = cv2.projectPoints(np.asarray(tG_fill[:,0:3],dtype=float),rotation_vector,\
                        translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)

    tGp_support = cv2.projectPoints(np.asarray(tG_support[:,0:3],dtype=float),rotation_vector,\
                        translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)
    #tGp_support_interface = cv2.projectPoints(np.asarray(tG_support_interface[:,0:3],dtype=float),\
                    #rotation_vector,translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)
    #-----------------------
    tGp_wall_inner_top_aux = cv2.projectPoints(np.asarray(tG_wall_inner_top_aux[:,0:3],dtype=float),\
                    rotation_vector,translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)
    tGp_wall_inner_bot_aux = cv2.projectPoints(np.asarray(tG_wall_inner_bot_aux[:,0:3],dtype=float),\
                    rotation_vector,translation_vector,camera_intrinsic_K,dist_coeffs)[0].reshape(-1, 2)
        
        
    for i in range(np.shape(tGp_default)[0]):
        plt.plot([tGp_default[i][0],tGp_default[i-1][0]],[tGp_default[i][1],tGp_default[i-1][1]],color='sienna')
    for i in range(np.shape(tGp_wall_outer)[0]):
        plt.plot([tGp_wall_outer[i][0],tGp_wall_outer[i-1][0]],\
             [tGp_wall_outer[i][1],tGp_wall_outer[i-1][1]],color='deepskyblue',linewidth=4)
    for i in range(np.shape(tGp_wall_inner)[0]):
        plt.plot([tGp_wall_inner[i][0],tGp_wall_inner[i-1][0]],\
                 [tGp_wall_inner[i][1],tGp_wall_inner[i-1][1]],color='tomato')
    for i in range(np.shape(tGp_fill)[0]):
        plt.plot([tGp_fill[i][0],tGp_fill[i-1][0]],[tGp_fill[i][1],tGp_fill[i-1][1]],color='aquamarine')
    for i in range(np.shape(tGp_support)[0]):
        plt.plot([tGp_support[i][0],tGp_support[i-1][0]],[tGp_support[i][1],tGp_support[i-1][1]],color='yellow')
    #-----------------------
    # uncomment to use the aux trajectories (+1, -1 layer height)
    '''for i in range(np.shape(tGp_wall_inner_top_aux)[0]):
        plt.plot([tGp_wall_inner_top_aux[i][0],tGp_wall_inner_top_aux[i-1][0]],\
                 [tGp_wall_inner_top_aux[i][1],tGp_wall_inner_top_aux[i-1][1]],color='pink')
    for i in range(np.shape(tGp_wall_inner_bot_aux)[0]):
        plt.plot([tGp_wall_inner_bot_aux[i][0],tGp_wall_inner_bot_aux[i-1][0]],\
                 [tGp_wall_inner_bot_aux[i][1],tGp_wall_inner_bot_aux[i-1][1]],color='hotpink')'''
plt.title("Segmented G-Code overlay | Image layer={} | G-Code layer={}".format(LAYER_NUMBER,LAYER_NUMBER-2))
plt.xlim(400,1000)
plt.ylim(450,120)
plt.show()


print(X_active_wall_inner)