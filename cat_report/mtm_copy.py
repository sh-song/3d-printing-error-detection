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

# Plot separate segmented G-Code trajectories for the single ayer

fig = plt.figure(figsize=(14,6), dpi=80)
ax = fig.add_subplot(121, projection='3d')
for i in range(len(X_active_default)):
    ax.plot([X_active_default[i],X_active_default[i-1]],
        [Y_active_default[i],Y_active_default[i-1]],
        [Z_active_default[i],Z_active_default[i-1]],color='sienna')

for i in range(len(X_active_wall_outer)):
    ax.plot([X_active_wall_outer[i],X_active_wall_outer[i-1]],
        [Y_active_wall_outer[i],Y_active_wall_outer[i-1]],
        [Z_active_wall_outer[i],Z_active_wall_outer[i-1]],color='royalblue')

for i in range(len(X_active_wall_inner)):
    ax.plot([X_active_wall_inner[i],X_active_wall_inner[i-1]],
        [Y_active_wall_inner[i],Y_active_wall_inner[i-1]],
        [Z_active_wall_inner[i],Z_active_wall_inner[i-1]],color='tomato')

for i in range(len(X_active_fill)):
    ax.plot([X_active_fill[i],X_active_fill[i-1]],
        [Y_active_fill[i],Y_active_fill[i-1]],
        [Z_active_fill[i],Z_active_fill[i-1]],color='lime')

for i in range(len(X_active_support)):
    ax.plot([X_active_support[i],X_active_support[i-1]],
        [Y_active_support[i],Y_active_support[i-1]],
        [Z_active_support[i],Z_active_support[i-1]],color='orange')

for i in range(len(X_active_support_interface)):
    ax.plot([X_active_support_interface[i],X_active_support_interface[i-1]],
        [Y_active_support_interface[i],Y_active_support_interface[i-1]],
        [Z_active_support_interface[i],Z_active_support_interface[i-1]],color='cyan')
ax.set_xlabel('X, mm')
ax.set_ylabel('Y, mm')
ax.set_title('Segmented G-Code | Image layer={} | G-Code Layer={}'.format(LAYER_NUMBER,LAYER_NUMBER-2))


ax = fig.add_subplot(122)
for i in range(len(X_active_default)):
    ax.plot([X_active_default[i],X_active_default[i-1]],
        [Y_active_default[i],Y_active_default[i-1]],color='sienna')

for i in range(len(X_active_wall_outer)):
    ax.plot([X_active_wall_outer[i],X_active_wall_outer[i-1]],
        [Y_active_wall_outer[i],Y_active_wall_outer[i-1]],color='royalblue')

for i in range(len(X_active_wall_inner)):
    ax.plot([X_active_wall_inner[i],X_active_wall_inner[i-1]],
        [Y_active_wall_inner[i],Y_active_wall_inner[i-1]],color='tomato')

for i in range(len(X_active_fill)):
    ax.plot([X_active_fill[i],X_active_fill[i-1]],
        [Y_active_fill[i],Y_active_fill[i-1]],color='lime')

for i in range(len(X_active_support)):
    ax.plot([X_active_support[i],X_active_support[i-1]],
        [Y_active_support[i],Y_active_support[i-1]],color='gold')

for i in range(len(X_active_support_interface)):
    ax.plot([X_active_support_interface[i],X_active_support_interface[i-1]],
        [Y_active_support_interface[i],Y_active_support_interface[i-1]],color='cyan')
ax.set_xlabel('X, mm')
ax.set_ylabel('Y, mm')
ax.set_title('G-Code trajectories for the Layer {}'.format(LAYER_NUMBER-2))
ax.grid(False)
ax.set_aspect(1)


plt.show()


# Here we create two masks (outer and inner) from the gcode type:wall-inner
# then substact "inner" from outer" to get the gcode-precise outline

gcode_mask_outer = np.zeros((src.shape), dtype=np.uint8)
cv2.polylines(gcode_mask_outer,[np.asarray(tGp_wall_inner,dtype=int)],False,(200,200,200),5)

gcode_mask_inner = np.zeros((src.shape), dtype=np.uint8)
cv2.polylines(gcode_mask_inner,[np.asarray(tGp_wall_inner,dtype=int)],False,(200,200,200),4)

### creating additional contours for the layers before and after
gcode_mask_outer_top = np.zeros((src.shape), dtype=np.uint8)
gcode_mask_inner_top = np.zeros((src.shape), dtype=np.uint8)
cv2.polylines(gcode_mask_outer_top,[np.asarray(tGp_wall_inner_top_aux,dtype=int)],False,(200,200,200),5)
cv2.polylines(gcode_mask_inner_top,[np.asarray(tGp_wall_inner_top_aux,dtype=int)],False,(200,200,200),4)

gcode_mask_outer_bot = np.zeros((src.shape), dtype=np.uint8)
gcode_mask_inner_bot = np.zeros((src.shape), dtype=np.uint8)
cv2.polylines(gcode_mask_outer_bot,[np.asarray(tGp_wall_inner_bot_aux,dtype=int)],False,(200,200,200),5)
cv2.polylines(gcode_mask_inner_bot,[np.asarray(tGp_wall_inner_bot_aux,dtype=int)],False,(200,200,200),4)

print(np.shape(src))
print(np.shape(gcode_mask_outer))
print(np.shape(gcode_mask_inner))


# Generating a precise 1px outline for the outer edge

final_mask = gcode_mask_outer*gcode_mask_inner
contours, hierarchy = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

final_mask_top = gcode_mask_outer_top*gcode_mask_inner_top
final_mask_bot = gcode_mask_outer_bot*gcode_mask_inner_bot
contours_top, hierarchy_top = cv2.findContours(final_mask_top, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_bot, hierarchy_bot = cv2.findContours(final_mask_bot, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



# Get the points from the detected opencv contours

# ------------------------------------------- Outer Shell
img = np.zeros((src.shape), np.uint8)

height = np.shape(final_mask)[0]
width = np.shape(final_mask)[1]

noOfCoordinates_t = 0
coordX_t = []
coordY_t = []
edgeMagnitude_t = []
edgeDerivativeX_t = []
edgeDerivativeY_t = []

shell = np.zeros((height,width,1), dtype=np.float32)
shell = np.squeeze(shell)

RSum = 0
CSum = 0

# draw contour 0
cv2.drawContours(img, contours, 0, (255,255,255), 1)

for i in range(0,width):
    for j in range(0,height):
        if (img[j][i] != 0):
            RSum = RSum+i
            CSum = CSum+j
            coordX_t.append(i)
            coordY_t.append(j)
            noOfCoordinates_t += 1
            shell[j][i]=1

center_gravity_x_outer = RSum/(noOfCoordinates_t+1e-6)
center_gravity_y_outer = CSum/(noOfCoordinates_t+1e-6)

outer_shell=np.vstack([(coordX_t),(coordY_t)])



# ------------------------- Inner Shell TOP (auxiliary contour to double check the max error gap)
img_top_aux = np.zeros((src.shape), np.uint8)

shell_top_aux = np.zeros((height,width,1), dtype=np.float32)
shell_top_aux = np.squeeze(shell_top_aux)

cv2.drawContours(img_top_aux, contours_top, 0, (255,255,255), 1)

coordX_t = []
coordY_t = []

for i in range(0,width):
    for j in range(0,height):
        if (img_top_aux[j][i] != 0):
            coordX_t.append(i)
            coordY_t.append(j)
            shell_top_aux[j][i]=1

outer_shell_top_aux = np.vstack([(coordX_t),(coordY_t)])


# ------------------------- Inner Shell BOTTOM  (auxiliary contour to double check the max error gap)
img_bot_aux = np.zeros((src.shape), np.uint8)

shell_bot_aux = np.zeros((height,width,1), dtype=np.float32)
shell_bot_aux = np.squeeze(shell_bot_aux)

cv2.drawContours(img_bot_aux, contours_bot, 0, (255,255,255), 1)

coordX_t = []
coordY_t = []

for i in range(0,width):
    for j in range(0,height):
        if (img_bot_aux[j][i] != 0):
            coordX_t.append(i)
            coordY_t.append(j)
            shell_bot_aux[j][i]=1

outer_shell_bot_aux = np.vstack([(coordX_t),(coordY_t)])



# ------------------------------------------- Inner Shell
img = np.zeros((src.shape), np.uint8)

height = np.shape(final_mask)[0]
width = np.shape(final_mask)[1]

noOfCoordinates_t = 0
coordX_t = []
coordY_t = []
edgeMagnitude_t = []
edgeDerivativeX_t = []
edgeDerivativeY_t = []

shell = np.zeros((height,width,1), dtype=np.float32)
shell = np.squeeze(shell)

RSum = 0
CSum = 0

# draw contour 1
if(np.shape(contours)[0]==2):
    cv2.drawContours(img, contours, 1, (255,255,255), 1)
elif(np.shape(contours)[0]==3):
    cv2.drawContours(img, contours, 2, (255,255,255), 1)

for i in range(0,height):
    for j in range(0,width):
        if (img[i][j] != 0):
            RSum = RSum+j
            CSum = CSum+i
            coordX_t.append(j)
            coordY_t.append(i)
            noOfCoordinates_t += 1
            shell[i][j]=1

center_gravity_x_inner = RSum/(noOfCoordinates_t+1e-6)
center_gravity_y_inner = CSum/(noOfCoordinates_t+1e-6)

inner_shell=np.vstack([(coordX_t),(coordY_t)])


# Plot detected layer outline plus two auxiliary (+1 up and -1 down) layer contours

fig = plt.figure(figsize=(14,14), dpi=80)
plt.imshow(src,cmap='gray')
plt.scatter(outer_shell[0],outer_shell[1],s=2,color='cyan',marker='s')
plt.scatter(center_gravity_x_outer,center_gravity_y_outer,s=250,marker='+',color='cyan')

plt.scatter(outer_shell_top_aux[0],outer_shell_top_aux[1],s=1,color='deeppink',marker='s') # +1 layer
plt.scatter(outer_shell_bot_aux[0],outer_shell_bot_aux[1],s=1,color='royalblue',marker='s') # -1 layer

plt.scatter(inner_shell[0],inner_shell[1],s=2,color='cyan',marker='.')
plt.scatter(center_gravity_x_inner,center_gravity_y_inner,s=50,marker='o',color='cyan')
plt.xlim(500,900)
plt.ylim(380,170)
plt.title("-1/0/+1 layer fork | Image layer={} | G-Code layer={}".format(LAYER_NUMBER,LAYER_NUMBER-2))
plt.show()

src_temp = src.copy()
#cv2.drawContours(src_temp, contours_top, 0, (255,255,255), 1)
#cv2.drawContours(src_temp, contours_bot, 0, (255,255,255), 1)
cv2.drawContours(src_temp, contours, 0, (255,255,255), 1)

# Pc = Projection coordinates
Pc = np.zeros((np.shape(outer_shell)[1],4),dtype=np.float32)
shell = np.vstack([(outer_shell[0]),(outer_shell[1])])
# z_level = LAYER_THICKNESS*(LAYER_NUMBER-2) ### defined in the beginning
print("z level = {} mm".format(z_level))

Pc[:,0] = shell[0] # x
Pc[:,1] = shell[1] # y
Pc[:,2] = z_level*np.ones((np.shape(outer_shell)[1],1),dtype=np.float32).T # z
Pc[:,3] = np.ones((np.shape(outer_shell)[1],1),dtype=np.float32).T # homogeneous coordinates


# visibility check
# Solve the system of equations 
# xmin * k + 1 * b = ymin and 
# xmax * k + 1 * b = ymax:
eq_coeffs = np.array([[np.min(Pc[:,0]),1], [np.max(Pc[:,0]),1]])
eq_answers = np.array([Pc[np.argmin(Pc[:,0])][1],Pc[np.argmax(Pc[:,0])][1]])
slope = np.linalg.solve(eq_coeffs, eq_answers)[0]
shift = np.linalg.solve(eq_coeffs, eq_answers)[1]


# Generate a 60-pixel wide wrapped pseudo-side-view image

side_view_window_shift = 20
side_view_window_width = 40

fig = plt.figure(figsize=(18,14), dpi=80)
plt.imshow(src,cmap='gray')
plt.scatter(391,221,c='springgreen',s=30)
plt.scatter(788,77,c='springgreen',s=30)
plt.scatter(1031,368,c='springgreen',s=30)
plt.scatter(569,562,c='springgreen',s=30)

plt.plot([391,788],[221,77],c='springgreen')
plt.plot([788,1031],[77,368],c='springgreen')
plt.plot([1031,569],[368,562],c='springgreen')
plt.plot([569,391],[562,221],c='springgreen')

#plt.plot([391,1031],[221,368],c='springgreen',linestyle=':')
#plt.plot([788,569],[77,562],c='springgreen',linestyle=':')

plt.plot([np.min(Pc[:,0]),np.max(Pc[:,0])],\
                 [Pc[np.argmin(Pc[:,0])][1],Pc[np.argmax(Pc[:,0])][1]],c='whitesmoke',linewidth=4,linestyle=':')
plt.scatter(np.min(Pc[:,0]),Pc[np.argmin(Pc[:,0])][1],color='gold',marker='s',s=60)
plt.scatter(np.max(Pc[:,0]),Pc[np.argmax(Pc[:,0])][1],color='gold',marker='s',s=60)

visible_edge_x = []
visible_edge_y_top = []
visible_edge_y_bottom = []

for i in range(np.shape(Pc)[0]):
    if(Pc[i][1] > slope*Pc[i][0]+shift):
        plt.scatter(Pc[i][0],Pc[i][1],color='dodgerblue',s=16)
        visible_edge_x.append(Pc[i][0])
        visible_edge_y_top.append(Pc[i][1]-side_view_window_shift)
        visible_edge_y_bottom.append(Pc[i][1]+side_view_window_width)
    else:
        plt.scatter(Pc[i][0],Pc[i][1],color='red',s=10)
        
plt.plot([np.min(Pc[:,0]),np.max(Pc[:,0])],\
                 [Pc[np.argmin(Pc[:,0])][1],Pc[np.argmax(Pc[:,0])][1]],c='whitesmoke',linewidth=4,linestyle=':')
plt.scatter(np.min(Pc[:,0]),Pc[np.argmin(Pc[:,0])][1],color='gold',marker='s',s=100)
plt.scatter(np.max(Pc[:,0]),Pc[np.argmax(Pc[:,0])][1],color='gold',marker='s',s=100)

plt.xlim(500,900)
plt.ylim(380,170)
plt.title("Printed layer visibility check | Image layer={} | G-Code layer={}".format(LAYER_NUMBER,LAYER_NUMBER-2))
plt.show()


# Remove contour "foldings" inside the visible region

mod_visible_edge_x = []
mod_visible_edge_y_top = []
mod_visible_edge_y_bot = []

fig = plt.figure(figsize=(14,6), dpi=80)

for i in range(np.shape(visible_edge_x)[0]):
    plt.scatter(visible_edge_x[i],300-visible_edge_y_top[i],c='k',s=5)
    plt.scatter(visible_edge_x[i],300-visible_edge_y_bottom[i],c='b',s=1)
    #plt.plot([visible_edge_x[i],visible_edge_x[i-1]],[300-visible_edge_y_top[i],300-visible_edge_y_top[i-1]],color='pink')

    x = visible_edge_x[i]
    y = visible_edge_y_top[i]
    
    if(visible_edge_x.count(visible_edge_x[i])>1):
        idx_of_multiple_xx_instances = [idx for idx, elem in enumerate(visible_edge_x) if elem == visible_edge_x[i]]
        yy_elements_of_multiple_xx = [visible_edge_y_top[idx].tolist() for idx in idx_of_multiple_xx_instances]
        x = visible_edge_x[i]
        y = np.max(yy_elements_of_multiple_xx)
    
    mod_visible_edge_x.append(x)
    mod_visible_edge_y_top.append(y)
    mod_visible_edge_y_bot.append(y+side_view_window_shift+side_view_window_width)
    
    plt.scatter(x,300-y,c='deeppink',s=20)
    plt.scatter(x,300-(y+side_view_window_shift+side_view_window_width),c='royalblue',s=20)
    
for i in range(len(mod_visible_edge_x)):
    plt.plot([mod_visible_edge_x[i],mod_visible_edge_x[i]],\
             [300-mod_visible_edge_y_top[i],300-mod_visible_edge_y_bot[i]],linestyle='-',color='silver')
    
plt.title("Visible edge | Image layer={} | G-Code layer={}".format(LAYER_NUMBER,LAYER_NUMBER-2))
plt.show()


mask = np.zeros(src.shape, dtype=np.uint8)

# points format: [[(10,10), (300,300), (10,300)]]
first = [tuple(map(tuple,np.stack((mod_visible_edge_x,mod_visible_edge_y_top)).T[np.stack((mod_visible_edge_x,mod_visible_edge_y_top)).T[:,0].argsort()]))] # stl edge
second = [tuple(map(tuple,np.flipud(np.stack((mod_visible_edge_x,mod_visible_edge_y_bot)).T[np.stack((mod_visible_edge_x,mod_visible_edge_y_bot)).T[:,0].argsort()])))] # shifted edge

# 'area' - is the visible curved side-view region cleared from additional inner cross-foldings
area = [tuple(map(tuple, np.squeeze(np.concatenate((first,second),axis=1))))]

roi_corners = np.array(area, dtype=np.int32)
cv2.fillPoly(mask, roi_corners, (255, 255, 255))

# apply the mask
masked_image = cv2.bitwise_and(src, mask)

# Plot the visible curved side-view region cleared from additional inner cross-foldings

fig = plt.figure(figsize=(18,14), dpi=80)
plt.imshow(src_temp,cmap='gray')
plt.imshow(masked_image,alpha=0.5,cmap='gray')

for i in range(np.shape(area)[1]):
    plt.scatter(area[0][i][0],area[0][i][1],color='springgreen',s=30)
    plt.plot([area[0][i][0],area[0][i-1][0]],[area[0][i][1],area[0][i-1][1]],color='springgreen',linewidth=2)
plt.xlim(500,900)
plt.ylim(400,170)
plt.title("Folded edge to unwrap | Image layer={} | G-Code layer={}".format(LAYER_NUMBER,LAYER_NUMBER-2))
plt.show()

# 'Cut' the wrapped region from the source image

mask_edge_x = []
mask_edge_y = []

for i in range(np.shape(mask)[1]): # width
    for j in range(np.shape(mask)[0]): # height
        if (mask[j][i]==255):
            mask_edge_x.append(i)
            mask_edge_y.append(j)
            break
        
        
        # Generate the zeroed unwrapped placeholder

slice_height = side_view_window_shift+side_view_window_width
unwrapped = np.zeros((slice_height,np.max(mask_edge_x)-np.min(mask_edge_x)),dtype=int)
print(np.shape(unwrapped))



# Unwrap the selected region of interest and calculate pixel-wise vertical errors

for i in range(np.max(mask_edge_x)-np.min(mask_edge_x)):
    unwrapped[:,i:i+1]=src[mask_edge_y[i]-side_view_window_shift:mask_edge_y[i]+side_view_window_width,\
                             np.min(mask_edge_x)+i:np.min(mask_edge_x)+i+1]
    
fig = plt.figure(figsize=(14,10), dpi=80)
plt.imshow(unwrapped,cmap='gray')
plt.title(' ')
# Experimentally have been found that 1 layer is equal to 2 pixels height
plt.axhline(y=36,linewidth=2, color='r',linestyle='-')
plt.axhline(y=44,linewidth=2, color='r',linestyle='-')
plt.axhline(y=40,linewidth=2, color='yellow',linestyle='-')
#plt.grid()
plt.title("+/- 2 layers band (4 layers in total)")
plt.show()

kernel = 9
src_denoised = filters.median(np.asarray((unwrapped),dtype=np.uint8),selem=np.ones((kernel,kernel)))
edges = cv2.Canny(src_denoised,20,50)

fig = plt.figure(figsize=(14,8), dpi=80)
plt.imshow(src_denoised)
plt.axhline(y=40,linewidth=3, color='r',linestyle=':')
plt.title("Filtered unwrapped visible edge | Image layer={} | G-Code layer={}".format(LAYER_NUMBER,LAYER_NUMBER-2))
plt.show()

fig = plt.figure(figsize=(14,8), dpi=80)
plt.imshow(edges,cmap="viridis")
plt.axhline(y=40,linewidth=3, color='r',linestyle=':')
plt.title("Edge pattern for visible edge | Image layer={} | G-Code layer={}".format(LAYER_NUMBER,LAYER_NUMBER-2))
plt.show()



# Visualize the pixel-wise vertical errors

REFERENCE_LAYER_LEVEL = 39
error_array = []

cumulative_vertical_level_error = 0
relative_vertical_level_error = 0

fig = plt.figure(figsize=(14,14), dpi=80)
#ax = plt.subplots()
#plt.imshow(np.zeros(edges.shape),cmap='gray',alpha=0.1)
plt.imshow(edges,cmap='gray',alpha=0.3)
plt.axhline(y=REFERENCE_LAYER_LEVEL,linewidth=4, color='gold',linestyle='-')

for i in range(edges.shape[1]): # horizontal
    all_vertical_edge_locations_per_x_column = [idx for idx, elem in enumerate(edges[:,i:i+1]) if elem == 255]
    if all_vertical_edge_locations_per_x_column != []:
        all_vertical_level_errors = [j - REFERENCE_LAYER_LEVEL for j in all_vertical_edge_locations_per_x_column]
    elif all_vertical_edge_locations_per_x_column == []:
        all_vertical_level_errors = REFERENCE_LAYER_LEVEL
    #print(all_vertical_edge_locations_per_x_column)
    vertical_level_error = np.min(np.abs(all_vertical_level_errors))
    error_array.append(vertical_level_error)
    #print(vertical_level_error)
    
    cumulative_vertical_level_error = cumulative_vertical_level_error+vertical_level_error
    
    plt.scatter(i,REFERENCE_LAYER_LEVEL-vertical_level_error,c='blue',s=100)
# 4 pixel height equals to two layers
plt.axhline(y=35,linewidth=2, color='red',linestyle='-')
plt.axhline(y=43,linewidth=2, color='red',linestyle='-')
#ax.set_aspect(aspect=0.5)
#plt.xlim(70,140)
#plt.ylim(50,30)
plt.show()


fig = plt.figure(figsize=(14,4), dpi=80)
plt.subplot(121)
plt.plot(error_array)
plt.title("Vertical error distribution")
plt.subplot(122)
plt.boxplot(error_array)
plt.axhline(y=2,linewidth=1, color='b',linestyle=':') # one layer
plt.axhline(y=4,linewidth=1, color='r',linestyle=':') # two layers
plt.title("LAYER_NUMBER = {}".format(LAYER_NUMBER))
plt.ylim(-1,42)
plt.show()
plt.show()


layer_width = edges.shape[1]
relative_vertical_level_error = cumulative_vertical_level_error/layer_width

print("\nSide view height validation data")
print("--------------------------------------------------------------------------------")
print('Layer width = {} px'.format(layer_width))
print('Total error of the vertical level = {} px'.format(cumulative_vertical_level_error))
print('Relative vertical error = {} (Total error / Layer width)'.format(relative_vertical_level_error))

print("\nMean error = {} px".format(np.mean(error_array)))
print("Median = {} px".format(np.median(error_array)))
print("Standard deviation = {} px".format(np.std(error_array)))
print("--------------------------------------------------------------------------------\n")










# Read the rectified top view image

if(LAYER_NUMBER<10):
    rect_top_img = cv2.imread('dataset/rect_crop/rect_crop_L0%d.jpg' %LAYER_NUMBER, 0)
else:
    rect_top_img = cv2.imread('dataset/rect_crop/rect_crop_L%d.jpg' %LAYER_NUMBER, 0)
    
    # Get mask from STL (at this precision level STL outline works as well as G-Code contour)

# Reference code from the `meshcut` Python library (MIT License)
# ------------------------------------------------------------------------
def points3d(verts, point_size=3, **kwargs):
    if 'mode' not in kwargs:
        kwargs['mode'] = 'point'
    p = mlab.points3d(verts[:, 0], verts[:, 1], verts[:, 2], **kwargs)
    p.actor.property.point_size = point_size


def trimesh3d(verts, faces, **kwargs):
    mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces,
                         **kwargs)


def orthogonal_vector(v):
    """Return an arbitrary vector that is orthogonal to v"""
    if v[1] != 0 or v[2] != 0:
        c = (1, 0, 0)
    else:
        c = (0, 1, 0)
    return np.cross(v, c)


def show_plane(orig, n, scale=1.0, **kwargs):
    """
    Show the plane with the given origin and normal. scale give its size
    """
    b1 = orthogonal_vector(n)
    b1 /= la.norm(b1)
    b2 = np.cross(b1, n)
    b2 /= la.norm(b2)
    verts = [orig + scale*(-b1 - b2),
             orig + scale*(b1 - b2),
             orig + scale*(b1 + b2),
             orig + scale*(-b1 + b2)]
    faces = [(0, 1, 2), (0, 2, 3)]
    trimesh3d(np.array(verts), faces, **kwargs)
# ------------------------------------------------------------------------

m = stl.mesh.Mesh.from_file('misc_data/70mm_low_poly_fox_MatterControl.stl')

# Flatten vertices array
verts = m.vectors.reshape(-1, 3)
# Generate corresponding faces array
faces = np.arange(len(verts)).reshape(-1, 3)

verts, faces = meshcut.merge_close_vertices(verts, faces)

obtained_mesh = mesh.Mesh.from_file('misc_data/70mm_low_poly_fox_MatterControl.stl')

volume,cog,inertia = obtained_mesh.get_mass_properties()
print("Volume                                  = {0}".format(volume))
print("Position of the center of gravity (COG) = {0}".format(cog))
print("Inertia matrix at expressed at the COG  = {0}".format(inertia[0,:]))
print("                                          {0}".format(inertia[1,:]))
print("                                          {0}".format(inertia[2,:]))

mesh_plane = meshcut.TriangleMesh(verts,faces)

plane_orig_1 = (0, 0, z_level)
plane_norm_1 = (0, 0, 1)
plane_norm_1 /= la.norm(plane_norm_1)

stl_plane_1 = meshcut.Plane(plane_orig_1, plane_norm_1)

P0 = meshcut.cross_section_mesh(mesh_plane,meshcut.Plane((0, 0, 0), (0, 0, 1)))
P1 = meshcut.cross_section_mesh(mesh_plane,stl_plane_1)

# Double check the height of the model
part_height = np.max(obtained_mesh.vectors[:,:,2])-np.min(obtained_mesh.vectors[:,:,2])
print('Total height of the part = {} mm'.format(part_height))

# Add additional points to STL outline to get a dense contour

stl_outline_x = []
stl_outline_y = []
number_of_additional_points = 10

def interpolatePoints(segment_one, segment_two, additional_points_in_between):
    return zip(np.linspace(segment_one[0], segment_two[0], additional_points_in_between+1),\
               np.linspace(segment_one[1], segment_two[1], additional_points_in_between+1))

for i in range(np.shape(P1[0])[0]):
    for j in range(len(list(interpolatePoints((P1[0][i][0],P1[0][i][1]),(P1[0][i-1][0],P1[0][i][1]),\
                                                 number_of_additional_points)))):
        xx = list(interpolatePoints((P1[0][i-1][0],P1[0][i-1][1]),(P1[0][i][0],P1[0][i][1]),\
                                       number_of_additional_points))[j][0]
        yy = list(interpolatePoints((P1[0][i-1][0],P1[0][i-1][1]),(P1[0][i][0],P1[0][i][1]),\
                                       number_of_additional_points))[j][1]
        stl_outline_x.append(xx)
        stl_outline_y.append(yy)
    stl_outline_x.append(P1[0][i][0])
    stl_outline_y.append(P1[0][i][1])

stl_out_shape = np.vstack([(stl_outline_x),(stl_outline_y)])
stl_out_shape = np.vstack([(stl_out_shape[0]),(stl_out_shape[1])])


fig = plt.figure(figsize=(8,8), dpi=80)
plt.subplot(111)

for i in range(len(X_active_wall_inner)):
    #plt.scatter(X_active_wall_inner[i],Y_active_wall_inner[i],marker='s',c='k',s=10,alpha=1)
    plt.plot([X_active_wall_inner[i],X_active_wall_inner[i-1]],
        [Y_active_wall_inner[i],Y_active_wall_inner[i-1]],color='salmon',linewidth=3,alpha=1)
    
'''for i in range(len(X_active_default)):
    plt.plot([X_active_default[i],X_active_default[i-1]],
        [Y_active_default[i],Y_active_default[i-1]],color='sienna',alpha=0.3)'''

'''for i in range(len(X_active_wall_outer)):
    plt.scatter(X_active_wall_outer[i],Y_active_wall_outer[i],marker='s',c='k',s=10,alpha=1)'''

for i in range(len(X_active_wall_outer)):
    plt.plot([X_active_wall_outer[i],X_active_wall_outer[i-1]],\
        [Y_active_wall_outer[i],Y_active_wall_outer[i-1]],color='dodgerblue',linewidth=7,alpha=1)

for i in range(len(X_active_fill)):
    plt.plot([X_active_fill[i],X_active_fill[i-1]],
        [Y_active_fill[i],Y_active_fill[i-1]],color='lime',linewidth=4,alpha=1)
    
'''for i in range(len(X_active_fill)):
    plt.scatter(X_active_fill[i],Y_active_fill[i],marker='s',c='k',s=10,alpha=1)'''

for i in range(len(X_active_support)):
    plt.plot([X_active_support[i],X_active_support[i-1]],
        [Y_active_support[i],Y_active_support[i-1]],color='gold',linewidth=5,alpha=1)

for i in range(np.shape(P1[0])[0]):
    for j in range(len(list(interpolatePoints((P1[0][i][0],P1[0][i][1]),(P1[0][i-1][0],P1[0][i][1]),\
                                                 number_of_additional_points)))):
        xx = list(interpolatePoints((P1[0][i-1][0],P1[0][i-1][1]),(P1[0][i][0],P1[0][i][1]),\
                                       number_of_additional_points))[j][0]
        yy = list(interpolatePoints((P1[0][i-1][0],P1[0][i-1][1]),(P1[0][i][0],P1[0][i][1]),\
                                       number_of_additional_points))[j][1]
        #plt.scatter(xx,yy,color='b',s=20)
        stl_outline_x.append(xx)
        stl_outline_y.append(yy)

    plt.plot([P1[0][i][0],P1[0][i-1][0]],[P1[0][i][1],P1[0][i-1][1]],color='k',linewidth=3,linestyle='--')
    plt.scatter(P1[0][i][0],P1[0][i][1],marker='s',color='k',s=25)
    stl_outline_x.append(P1[0][i][0])
    stl_outline_y.append(P1[0][i][1])
plt.scatter(0,0,c='k',s=1550,marker='+')
#plt.xlim(-15, 25)
#plt.ylim(-20, 20)
ax.set_aspect(1)
plt.xlabel('X, mm')
plt.ylabel('Y, mm')
plt.title('STL vs G-Code for the layer {}'.format(LAYER_NUMBER))
plt.show()


# Create mask from STL
# Apply additional transformation to the contour if necessary

def SRT(point_cloud,scale,theta,tx,ty):
    S = scale
    R = np.array([[np.cos(theta),np.sin(theta)],\
                  [-np.sin(theta),np.cos(theta)]])
    T = np.array([tx,ty])
    transformed = S*R.dot(point_cloud)+T[:,np.newaxis]
    return transformed


stl_out_shape = np.vstack([(stl_outline_x),(stl_outline_y)])
stl_out_shape = SRT(stl_out_shape,5.6,0.2,273,230)
stl_out_shape = np.vstack([(stl_out_shape[0]),(stl_out_shape[1])])

stl_mask_points = np.asarray(stl_out_shape.T.reshape((-1,2)),dtype=np.int32)
print(np.shape(stl_mask_points))


# Draw the STL mask for edge detection

STL_MASK_WIDTH = 30
stl_mask = np.zeros((rect_top_img.shape),np.uint8)
cv2.polylines(stl_mask,[stl_mask_points],False,(255,255,255),STL_MASK_WIDTH)

print(np.shape(rect_top_img))
print(np.shape(stl_mask))
#print(np.shape(gcode_mask))

fig = plt.figure(figsize=(14,5), dpi=80)
plt.subplot(131)
plt.imshow(rect_top_img[100:400,150:400],cmap='Greys')
plt.imshow(np.flip(stl_mask,0)[100:400,150:400],alpha=0.3,cmap='prism')
plt.title('Contour mask for layer {}'.format(LAYER_NUMBER))
plt.subplot(132)
plt.scatter(stl_outline_x,stl_outline_y,c='b')
plt.title("Original STL outline")
plt.xlabel("X, mm")
plt.ylabel("Y, mm")
plt.grid()
plt.subplot(133)
plt.scatter(stl_out_shape[0],stl_out_shape[1],color='r',s=5)
plt.title("Transformed STL outline")
plt.xlabel("X, px")
plt.ylabel("Y, px")
plt.grid()
plt.show()

# Apply median filtration to reduce amount of visual noise
filtered_top_view = filters.median(rect_top_img,selem=np.ones((9,9)))

# Find contours using Canny edge detector
# Experimentally has been found that the 20...50 threshold range works well
top_view_edges = cv2.Canny(filtered_top_view,20,50)*(np.flip(stl_mask,0))

# But it can be more convenient to use Canny edge detector with automatic 
# parameters introduced by Adrian Rosebrock 
# (https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

edges_new = auto_canny(filtered_top_view)*(np.flip(stl_mask,0))

fig = plt.figure(figsize=(14,5), dpi=80)
plt.subplot(131)
plt.imshow(filtered_top_view[100:400,150:400],cmap='Greys')
plt.title('Filtered top view, Layer {}'.format(LAYER_NUMBER))
plt.xlabel("X, px")
plt.ylabel("Y, px")
plt.subplot(132)
plt.imshow(top_view_edges[100:400,150:400],cmap='Greys')
plt.title("Canny with manual parameters")
plt.xlabel("X, px")
plt.ylabel("Y, px")
plt.grid()
plt.subplot(133)
plt.imshow(edges_new[100:400,150:400],cmap='Greys')
plt.title("Canny with auto parameters")
plt.xlabel("X, px")
plt.ylabel("Y, px")
plt.grid()
plt.show()

# Convert detected contour into array of points for further ICP algorithm

height = np.shape(edges_new)[0]
width = np.shape(edges_new)[1]

noOfCoordinates_t = 0
coordX_t = []
coordY_t = []
edgeMagnitude_t = []
edgeDerivativeX_t = []
edgeDerivativeY_t = []

pp = np.zeros((height,width,1), dtype=np.float32)
pp = np.squeeze(pp)

RSum = 0
CSum = 0

for j in range(0,height):
    for i in range(0,width):
        if (edges_new[j][i] != 0):
            RSum = RSum+i
            CSum = CSum+j
            coordX_t.append(i)
            coordY_t.append(j)
            noOfCoordinates_t += 1
            pp[i][j]=1

center_gravity_x = RSum/(noOfCoordinates_t+1e-6)
center_gravity_y = CSum/(noOfCoordinates_t+1e-6)

detected_masked_edge_points = np.vstack([(coordX_t),(coordY_t)])


# Generate a contour template for Multi Template Matching

# here we create two masks (outer and inner) from the gcode type:wall-inner
# then substact "inner" from outer" to get the gcode-precise outline

gcode_wall_inner_shape = np.vstack([(X_active_wall_inner),(Y_active_wall_inner)])
gcode_wall_inner_shape = SRT(gcode_wall_inner_shape, 5.7,0.2,273,231)
#gcode_wall_inner_shape = SRT(gcode_wall_inner_shape, 5.7,0.2,277,231)
gcode_wall_inner_mask_points = np.asarray(gcode_wall_inner_shape.T.reshape((-1,2)),dtype=np.int32)

gcode_mask_outer = np.zeros((rect_top_img.shape), np.uint8)
cv2.polylines(gcode_mask_outer,[gcode_wall_inner_mask_points],False,(200,200,200),6)

gcode_mask_inner = np.zeros((rect_top_img.shape), np.uint8)
cv2.polylines(gcode_mask_inner,[gcode_wall_inner_mask_points],False,(200,200,200),4)

print(np.shape(rect_top_img))
print(np.shape(gcode_mask_outer))
print(np.shape(gcode_mask_inner))

final_mask = np.flip(gcode_mask_outer,0)*np.flip(gcode_mask_inner,0)

fig = plt.figure(figsize=(14,10), dpi=80)
plt.subplot(121)
plt.imshow(filtered_top_view,cmap='gray')
plt.imshow(auto_canny(filtered_top_view),alpha=0.2,cmap='gray')
plt.scatter(detected_masked_edge_points[0],detected_masked_edge_points[1],s=2,c='yellow')
plt.xlim(100,400)
plt.ylim(400,100)
plt.title("Masked edges for layer {}".format(LAYER_NUMBER))
plt.subplot(122)
plt.imshow(rect_top_img,cmap='gray')
plt.imshow(np.flip(gcode_mask_outer,0)-np.flip(gcode_mask_inner,0),alpha=0.3,cmap='magma')
plt.xlim(100,400)
plt.ylim(400,100)
plt.title("Reference contour alignment")
plt.show()

# Find the part using MTM

template_to_match = np.flip(gcode_mask_outer,0)-np.flip(gcode_mask_inner,0)
template_to_match = template_to_match[120:390,142:412]
print(template_to_match.shape)

print(np.max(template_to_match))
print(np.max(auto_canny(filtered_top_view)))

fig = plt.figure(figsize=(14,10), dpi=80)
plt.subplot(121)
plt.imshow(auto_canny(filtered_top_view),cmap='gray')
plt.title('Failed source image')
plt.axis('off')
plt.subplot(122)
plt.imshow(template_to_match,cmap='gray')
plt.title('Template')
plt.tight_layout()
plt.axis('off')
plt.show()

# Format the template into a list of tuple (label, templateImage)
listTemplate = [("L"+str(LAYER_NUMBER),template_to_match)]

# Find a match template in the image, if any
Hits = matchTemplates(listTemplate,auto_canny(filtered_top_view),score_threshold=0.01,\
                      method=cv2.TM_CCOEFF_NORMED,maxOverlap=0)

print("Found {} hits".format( len(Hits.index) ) )
print(Hits)

Overlay = drawBoxesOnRGB(auto_canny(filtered_top_view),Hits,showLabel=False)

Overlay[Hits.BBox[0][1]:Hits.BBox[0][1]+Hits.BBox[0][2],\
        Hits.BBox[0][0]:Hits.BBox[0][0]+Hits.BBox[0][3]] = (80,100,100)


contours, hierarchy = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(np.shape(contours))

# Get the points from the detected opencv contours

# ------------------------------------------- Outer Shell
img = np.zeros((rect_top_img.shape), np.uint8)

height = np.shape(final_mask)[0]
width = np.shape(final_mask)[1]

noOfCoordinates_t = 0
coordX_t = []
coordY_t = []
edgeMagnitude_t = []
edgeDerivativeX_t = []
edgeDerivativeY_t = []

shell = np.zeros((height,width,1), dtype=np.float32)
shell = np.squeeze(shell)

RSum = 0
CSum = 0

# draw contour 0
cv2.drawContours(img, contours, 0, (255,255,255), 1)

for j in range(0,height):
    for i in range(0,width):
        if (img[j][i] != 0):
            RSum = RSum+i
            CSum = CSum+j
            coordX_t.append(i)
            coordY_t.append(j)
            noOfCoordinates_t += 1
            shell[i][j]=1

center_gravity_x_outer = RSum/(noOfCoordinates_t+1e-6)
center_gravity_y_outer = CSum/(noOfCoordinates_t+1e-6)

outer_shell=np.vstack([(coordX_t),(coordY_t)])

# ------------------------------------------- Inner Shell
img = np.zeros((rect_top_img.shape), np.uint8)

height = np.shape(final_mask)[0]
width = np.shape(final_mask)[1]

noOfCoordinates_t = 0
coordX_t = []
coordY_t = []
edgeMagnitude_t = []
edgeDerivativeX_t = []
edgeDerivativeY_t = []

shell = np.zeros((height,width,1), dtype=np.float32)
shell = np.squeeze(shell)

RSum = 0
CSum = 0

# draw contour 1
if (np.shape(contours)[0]==2):
    cv2.drawContours(img, contours, 1, (255,255,255), 1)
else:
    cv2.drawContours(img, contours, 2, (255,255,255), 1)

for j in range(0,height):
    for i in range(0,width):
        if (img[j][i] != 0):
            RSum = RSum+i
            CSum = CSum+j
            coordX_t.append(i)
            coordY_t.append(j)
            noOfCoordinates_t += 1
            shell[i][j]=1

center_gravity_x_inner = RSum/(noOfCoordinates_t+1e-6)
center_gravity_y_inner = CSum/(noOfCoordinates_t+1e-6)

inner_shell=np.vstack([(coordX_t),(coordY_t)])

detected_vert_shift = Hits.BBox[0][1]+Hits.BBox[0][3]/2 - center_gravity_y_outer
detected_hor_shift = Hits.BBox[0][0]+Hits.BBox[0][2]/2 - center_gravity_x_outer

print("Detected VERTICAL shift = {} px".format(detected_vert_shift))
print("Detected HORIZONTAL shift = {} px".format(detected_hor_shift))

fig = plt.figure(figsize=(14,14), dpi=80)
plt.imshow(auto_canny(filtered_top_view),cmap='gray')
plt.imshow(auto_canny(filtered_top_view),alpha=0.2,cmap='gray')
plt.imshow(Overlay,alpha=0.7)
plt.scatter(Hits.BBox[0][0]+Hits.BBox[0][2]/2,Hits.BBox[0][1]+Hits.BBox[0][3]/2,c='yellow',marker='+',s=750)
plt.scatter(Hits.BBox[0][0]+Hits.BBox[0][2]/2,Hits.BBox[0][1]+Hits.BBox[0][3]/2,c='yellow',marker='o',s=100)

plt.plot([center_gravity_x_outer,Hits.BBox[0][0]+Hits.BBox[0][2]/2],\
         [center_gravity_y_outer,Hits.BBox[0][1]+Hits.BBox[0][3]/2],c='yellow',linewidth=3,linestyle=':')

plt.scatter(outer_shell[0]+detected_hor_shift,\
            outer_shell[1]+detected_vert_shift,s=15,color='springgreen',label='Detected layer outline')
plt.scatter(inner_shell[0]+detected_hor_shift,\
            inner_shell[1]+detected_vert_shift,s=2,color='springgreen')

plt.scatter(outer_shell[0],outer_shell[1],s=10,color='deeppink',label='Reference layer outline')
plt.scatter(center_gravity_x_outer,center_gravity_y_outer,s=750,marker='+',color='deeppink')
plt.scatter(center_gravity_x_outer,center_gravity_y_outer,s=100,marker='o',color='deeppink')
plt.scatter(inner_shell[0],inner_shell[1],s=1,color='deeppink')
plt.title('Matched template for layer {}'.format(LAYER_NUMBER))
plt.legend()
plt.show()

Overlay = drawBoxesOnRGB(255*edges_new,Hits,showLabel=False)
#Overlay[100:150,120:150] = (255,250,20)

Overlay[Hits.BBox[0][1]:Hits.BBox[0][1]+Hits.BBox[0][2],\
        Hits.BBox[0][0]:Hits.BBox[0][0]+Hits.BBox[0][3]] = (80,85,20)
Overlay[Hits.BBox[0][1]:Hits.BBox[0][1]+Hits.BBox[0][2],\
        Hits.BBox[0][0]:Hits.BBox[0][0]+Hits.BBox[0][3],0] = template_to_match

fig = plt.figure(figsize=(14,10), dpi=80)
plt.imshow(filtered_top_view,cmap='gray')
plt.imshow(auto_canny(filtered_top_view),alpha=0.3,cmap='gray')
plt.scatter(detected_masked_edge_points[0],detected_masked_edge_points[1],s=6,c='royalblue',\
            label='Detected edge points')
plt.imshow(Overlay,alpha=0.5)
plt.scatter(Hits.BBox[0][0]+Hits.BBox[0][2]/2,Hits.BBox[0][1]+Hits.BBox[0][3]/2,c='yellow',marker='+',s=350)

plt.scatter(outer_shell[0],outer_shell[1],s=10,color='springgreen',label='Reference contour')
plt.imshow(np.flip(stl_mask,0),alpha=0.3,cmap='Blues')
#plt.scatter(center_gravity_x_outer,center_gravity_y_outer,s=250,marker='+',color='salmon')
#plt.scatter(inner_shell[0],inner_shell[1],s=5,color='deeppink')
plt.title("Combined plot (detected edges, reference contour, and mask) for the layer {}".format(LAYER_NUMBER))
plt.legend()
plt.xlim(50,500)
plt.ylim(400,100)
plt.show()

# Sparse the contours to speed up the ICP algorithm

sparse_shell_x = np.delete(outer_shell[0], np.arange(0, outer_shell[0].size, 3))
sparse_shell_y = np.delete(outer_shell[1], np.arange(0, outer_shell[1].size, 3))
# sparse more
sparse_shell_x = np.delete(sparse_shell_x, np.arange(0, sparse_shell_x.size, 2))
sparse_shell_y = np.delete(sparse_shell_y, np.arange(0, sparse_shell_y.size, 2))

print('Size reduction for reference contour: {} -> {} points'.format(outer_shell[0].size,sparse_shell_x.size))
sparse_outer_shell=np.vstack([(sparse_shell_x),(sparse_shell_y)])
##########################

sparse_p_x = np.delete(detected_masked_edge_points[0],np.arange(0,detected_masked_edge_points[0].size,3))
sparse_p_y = np.delete(detected_masked_edge_points[1],np.arange(0,detected_masked_edge_points[1].size,3))

# sparse more
sparse_p_x = np.delete(sparse_p_x, np.arange(0,sparse_p_x.size,2))
sparse_p_y = np.delete(sparse_p_y, np.arange(0,sparse_p_y.size,2))

print('Size reduction for detected contour: {} -> {} points'.format(detected_masked_edge_points[0].size,sparse_p_x.size))
sparse_points=np.vstack([(sparse_p_x),(sparse_p_y)])
##########################

# Compare the gcode points and the detected contour points

fig = plt.figure(figsize=(12,12), dpi=80)
plt.scatter(detected_masked_edge_points[0],detected_masked_edge_points[1],s=100,c='pink',\
            label='Initially detected contour points')
plt.scatter(sparse_points[0],sparse_points[1],s=6,c='red',label='Sparced contour points')
plt.scatter(outer_shell[0],outer_shell[1],s=100,color='lightblue',label='Initial reference contour points')
plt.scatter(sparse_outer_shell[0],sparse_outer_shell[1],s=6,color='blue',\
            label='Sparced reference contour points')

plt.scatter(center_gravity_x,center_gravity_y,s=200,marker='x',c='red',\
            label='Mass center of the detected contour')
plt.scatter(center_gravity_x_outer,center_gravity_y_outer,s=200,marker='+',color='blue',\
           label='Mass center of the reference contour')

plt.legend()
plt.grid()
plt.show()

# Iterative Closest Points algorithm

# During the iteration step we use previously updated RTS matrix
# and recalculate correspondence between two pointclouds
def iteration_ICP(RTS,reference_point_cloud,updated_point_cloud):
    global detected_theta_total
    global detected_tr_x_total
    global detected_tr_y_total
    global detected_scale
    global total_correspondence
    err_temp = 0
    
    R = RTS[0:2,0:2]
    T = RTS[0:2,2]
    updated_point_cloud = R.dot(updated_point_cloud)+T[:,np.newaxis]
    
    # Find correspondence-------------------------
    temp_ref_ptcd = []
    for i, point in enumerate(updated_point_cloud.T):
        distances = []
        for rp in reference_point_cloud.T:
            distance_map = point[:2]-rp[:2]
            distance_norm = np.linalg.norm(distance_map)
            distances.append(distance_norm)

        if distances[np.argmin(distances)] < 25:
            temp_ref_ptcd.append(reference_point_cloud.T[np.argmin(distances)])
            
    temp_ref_ptcd = np.array(temp_ref_ptcd).T
    temp_ptcd = updated_point_cloud.copy()
    #---------------------------------------------

    for (m,n) in zip(temp_ptcd.T, temp_ref_ptcd.T):
        err_temp = np.sum(np.sqrt((np.abs(n[0])-np.abs(m[0]))**2+\
                                  (np.abs(n[1])-np.abs(m[1]))**2),\
                          axis=0)
    total_correspondence.append(err_temp)
    
    R,T,S = update_RTS(temp_ref_ptcd,temp_ptcd)
    detected_scale = 1/S

    if (R[0][0]>1.0):
        detected_theta_total = detected_theta_total + np.arccos(1)*180/np.pi
    else:
        detected_theta_total = detected_theta_total + np.arccos(R[0][0])*180/np.pi
    
    detected_tr_x_total = detected_tr_x_total + T[0]
    detected_tr_y_total = detected_tr_y_total + T[1]
    
    upd_R = RTS[0:2,0:2]
    upd_T = RTS[0:2,2]
    upd_RTS = np.eye(3,dtype=float)
    upd_RTS[0:2,0:2] = R.dot(upd_R)
    upd_RTS[0:2,2] = R.dot(upd_T)+T
    upd_RTS[2,0:2] = np.zeros(2,dtype=float)
    print('...')
    return upd_RTS


# During the update RTS step we calculate current R,T,S parameters and use them
# as initial parameters for the following ICP iteration
def update_RTS(reference_pointcloud,detected_pointcloud):
    white_detected_points = detected_pointcloud.T-np.mean(detected_pointcloud.T, axis=0)
    white_reference_points = reference_pointcloud.T-np.mean(reference_pointcloud.T, axis=0)
    
    for_singular_values = np.dot(white_detected_points.T,white_reference_points)
    U,Sigma,V = np.linalg.svd(for_singular_values.T)
    
    # Compute rotation matrix
    R = np.dot(U,V)
    # Compute translation vector
    T = np.mean(reference_pointcloud.T, axis=0)-np.dot(R,np.mean(detected_pointcloud.T, axis=0))
    # Compute scaling factor
    S = np.sqrt((np.linalg.norm(white_reference_points))**2/(np.linalg.norm(white_detected_points))**2)
    return R,T,S


detected_theta_total = 0
detected_tr_x_total = 0
detected_tr_y_total = 0
detected_scale = 0
total_correspondence = []

# initial RTS transformation matrix
RTS_init = np.eye(3,dtype=float)

ICP_iterations = 5
for ICP_iteration in range(ICP_iterations):
    RTS_init = iteration_ICP(RTS_init,sparse_points,sparse_outer_shell)

print("-------------------------------------------------------")
print("Rotation detected: {} degrees".format(np.round(detected_theta_total,2)))
print("Translation detected: {} x/y units".format([np.round(-detected_tr_x_total,2),\
                                                   np.round(-detected_tr_y_total,2)]))
print("Detected scaling factor: {} %".format(np.round(detected_scale*100),2))
print("-------------------------------------------------------")

# Plot total correspondence between contour points over the ICP iterations

fig = plt.figure(figsize=(14,4), dpi=80)
plt.plot(total_correspondence)
plt.scatter(len(total_correspondence)-1,total_correspondence[-1])
plt.grid()
plt.title("Total correspondence over the ICP iterations")
plt.show()

print("Total correspondence = {} px". format(np.round(total_correspondence[-1],4)))