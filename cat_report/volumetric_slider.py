# Volumetric view

import numpy as np
import os
import sys
import glob
import time

from stl import mesh
from mpl_toolkits import mplot3d

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QComboBox, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import cv2
from PIL import Image
from skimage import filters
from skimage.io import imread, imshow, imsave, imread_collection, concatenate_images
from skimage.transform import resize


class App(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'Volumetric View'
        self.initUI()
 
    def initUI(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setWindowTitle(self.title)

        # ---------- COMPONENTS ---------------------------------------------------
        # -------- V0
        self.lbl_blank_space = QtWidgets.QLabel(' ')
        
        self.lbl_LOGO   = QtWidgets.QLabel('VOLUMETRIC VIEW')
        #self.lbl_LOGO.setPixmap(QtGui.QPixmap('logo_sensor_fusion2.png'))
        self.lbl_LOGO.setAlignment(Qt.AlignCenter)
        self.lbl_LOGO.setStyleSheet("QLabel {color: #333333;}")
        self.lbl_LOGO.setFont(QtGui.QFont('Sans',20,QtGui.QFont.Bold))
        self.lbl_LOGO.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.btn_LOAD_IMG = QtWidgets.QPushButton('Load image set')
        self.btn_LOAD_IMG.setStyleSheet("QPushButton {background-color: #999999;}")
        self.btn_LOAD_IMG.setFont(QtGui.QFont('Sans',14,QtGui.QFont.Bold))
        #self.btn_LOAD_STL = QtWidgets.QPushButton('Load STL file')

        self.lbl_SLICING  = QtWidgets.QLabel('XYZ Slicing')
        self.lbl_SLICING.setStyleSheet("QLabel {background-color: #333333; color: #FFFFFF}")
        self.lbl_SLICING.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))
        self.lbl_SLICING.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.lbl_X = QtWidgets.QLabel('X')
        self.lbl_X.setStyleSheet("QLabel {background-color: #EE7070;}")
        self.lbl_X.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))

        self.lbl_Y = QtWidgets.QLabel('Y')
        self.lbl_Y.setStyleSheet("QLabel {background-color: #70EE70;}")
        self.lbl_Y.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))

        self.lbl_Z = QtWidgets.QLabel('Z')
        self.lbl_Z.setStyleSheet("QLabel {background-color: #7070EE;}")
        self.lbl_Z.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))

        self.slider_X = QSlider(Qt.Horizontal)
        self.slider_X.setFocusPolicy(Qt.StrongFocus)
        self.slider_X.setTickPosition(QSlider.TicksAbove)
        self.slider_X.setTickInterval(2)
        self.slider_X.setSingleStep(1)

        self.slider_Y = QSlider(Qt.Horizontal)
        self.slider_Y.setFocusPolicy(Qt.StrongFocus)
        self.slider_Y.setTickPosition(QSlider.TicksAbove)
        self.slider_Y.setTickInterval(2)
        self.slider_Y.setSingleStep(1)

        self.slider_Z = QSlider(Qt.Horizontal)
        self.slider_Z.setFocusPolicy(Qt.StrongFocus)
        self.slider_Z.setTickPosition(QSlider.TicksAbove)
        self.slider_Z.setTickInterval(2)
        self.slider_Z.setSingleStep(1)

        self.lbl_WORKSPACE   = QtWidgets.QLabel('Workspace')
        self.lbl_WORKSPACE.setStyleSheet("QLabel {background-color: #333333; color: #FFFFFF}")
        self.lbl_WORKSPACE.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))
        self.lbl_WORKSPACE.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        workspace_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self._workspace_ax = workspace_canvas.figure.add_subplot(111, projection='3d')
        self._workspace_ax.grid()
        self._workspace_ax.set_title('3D Printer Workspace')
        self._workspace_ax.set_ylabel('Y')
        self._workspace_ax.set_xlabel('X')

        self.lbl_INSTANT_X   = QtWidgets.QLabel('X=')
        self.lbl_INSTANT_X.setStyleSheet("QLabel {background-color: #EE7070; color: #000000}")
        self.lbl_INSTANT_X.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))
        self.lbl_INSTANT_X.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.lbl_INSTANT_Y   = QtWidgets.QLabel('Y=')
        self.lbl_INSTANT_Y.setStyleSheet("QLabel {background-color: #70EE70; color: #000000}")
        self.lbl_INSTANT_Y.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))
        self.lbl_INSTANT_Y.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.lbl_INSTANT_Z   = QtWidgets.QLabel('Z=')
        self.lbl_INSTANT_Z.setStyleSheet("QLabel {background-color: #7070EE; color: #000000}")
        self.lbl_INSTANT_Z.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))
        self.lbl_INSTANT_Z.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        #self.btn_GENERATE_VOXEL_PLOT = QtWidgets.QPushButton('Generate Voxel Plot')

        # -------- V1
        self.lbl_XY_SLICE   = QtWidgets.QLabel('XY SLICE (Captured Frame)')
        self.lbl_XY_SLICE.setStyleSheet("QLabel {background-color: #333333; color: #FFFFFF}")
        self.lbl_XY_SLICE.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))
        self.lbl_XY_SLICE.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.lbl_SIDE_VIEW   = QtWidgets.QLabel('CONTOUR VIEW')
        self.lbl_SIDE_VIEW.setStyleSheet("QLabel {background-color: #333333; color: #FFFFFF}")
        self.lbl_SIDE_VIEW.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))
        self.lbl_SIDE_VIEW.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.lbl_XZ_SLICE   = QtWidgets.QLabel('XZ SLICE')
        self.lbl_XZ_SLICE.setStyleSheet("QLabel {background-color: #333333; color: #FFFFFF}")
        self.lbl_XZ_SLICE.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))
        self.lbl_XZ_SLICE.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.lbl_YZ_SLICE   = QtWidgets.QLabel('YZ SLICE')
        self.lbl_YZ_SLICE.setStyleSheet("QLabel {background-color: #333333; color: #FFFFFF}")
        self.lbl_YZ_SLICE.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))
        self.lbl_YZ_SLICE.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        XY_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self._XY_ax = XY_canvas.figure.subplots()
        self._XY_ax.grid()
        self._XY_ax.set_title('Captured Frame')
        self._XY_ax.set_ylabel('Y')
        self._XY_ax.set_xlabel('X')

        SIDE_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self._SIDE_ax = SIDE_canvas.figure.subplots()
        self._SIDE_ax.grid()
        self._SIDE_ax.set_title('CONTOUR VIEW')
        self._SIDE_ax.set_ylabel('Z')
        self._SIDE_ax.set_xlabel('X')

        XZ_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self._XZ_ax = XZ_canvas.figure.subplots()
        self._XZ_ax.grid()
        self._XZ_ax.set_title('XZ SLICE')
        self._XZ_ax.set_ylabel('Z')
        self._XZ_ax.set_xlabel('X')

        YZ_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self._YZ_ax = YZ_canvas.figure.subplots()
        self._YZ_ax.grid()
        self._YZ_ax.set_title('YZ SLICE')
        self._YZ_ax.set_ylabel('Z')
        self._YZ_ax.set_xlabel('Y')

        self._timer = XY_canvas.new_timer(100, [(self._update_canvas, (), {})])
        self._timer.start()

        # ---------- LAYOUT -------------------------------------------------------
        layout_MAIN_H = QtWidgets.QHBoxLayout()
        layout_V0 = QtWidgets.QVBoxLayout()
        layout_V1 = QtWidgets.QVBoxLayout()
        layout_H_X = QtWidgets.QHBoxLayout()
        layout_H_Y = QtWidgets.QHBoxLayout()
        layout_H_Z = QtWidgets.QHBoxLayout()
        layout_instant_XYZ = QtWidgets.QHBoxLayout()

        layout_H_XY_SIDE_lbl = QtWidgets.QHBoxLayout()
        layout_H_XY_SIDE_canvas = QtWidgets.QHBoxLayout()
        layout_H_XZ_YZ_lbl = QtWidgets.QHBoxLayout()
        layout_H_XZ_YZ_canvas = QtWidgets.QHBoxLayout()


        # -------- V0
        layout_V0.addWidget(self.lbl_LOGO)
        layout_V0.addWidget(self.lbl_blank_space)
        layout_V0.addWidget(self.btn_LOAD_IMG)
        #layout_V0.addWidget(self.btn_LOAD_STL)
        layout_V0.addWidget(self.lbl_blank_space)
        layout_V0.addWidget(self.lbl_SLICING)

        layout_H_X.addWidget(self.lbl_X)
        layout_H_X.addWidget(self.slider_X)
        layout_V0.addLayout(layout_H_X)

        layout_H_Y.addWidget(self.lbl_Y)
        layout_H_Y.addWidget(self.slider_Y)
        layout_V0.addLayout(layout_H_Y)

        layout_H_Z.addWidget(self.lbl_Z)
        layout_H_Z.addWidget(self.slider_Z)
        layout_V0.addLayout(layout_H_Z)

        layout_V0.addWidget(self.lbl_blank_space)
        layout_V0.addWidget(self.lbl_blank_space)
        layout_V0.addWidget(self.lbl_blank_space)
        layout_V0.addWidget(self.lbl_blank_space)
        layout_V0.addWidget(self.lbl_blank_space)

        layout_V0.addWidget(self.lbl_WORKSPACE)
        layout_V0.addWidget(workspace_canvas)

        layout_instant_XYZ.addWidget(self.lbl_INSTANT_X)
        layout_instant_XYZ.addWidget(self.lbl_INSTANT_Y)
        layout_instant_XYZ.addWidget(self.lbl_INSTANT_Z)
        layout_V0.addLayout(layout_instant_XYZ)

        # -------- V1
        layout_H_XY_SIDE_lbl.addWidget(self.lbl_XY_SLICE)
        layout_H_XY_SIDE_lbl.addWidget(self.lbl_SIDE_VIEW)
        layout_V1.addLayout(layout_H_XY_SIDE_lbl)

        layout_H_XY_SIDE_canvas.addWidget(XY_canvas)
        layout_H_XY_SIDE_canvas.addWidget(SIDE_canvas)
        layout_V1.addLayout(layout_H_XY_SIDE_canvas)

        layout_H_XZ_YZ_lbl.addWidget(self.lbl_XZ_SLICE)
        layout_H_XZ_YZ_lbl.addWidget(self.lbl_YZ_SLICE)
        layout_V1.addLayout(layout_H_XZ_YZ_lbl)

        layout_H_XZ_YZ_canvas.addWidget(XZ_canvas)
        layout_H_XZ_YZ_canvas.addWidget(YZ_canvas)
        layout_V1.addLayout(layout_H_XZ_YZ_canvas)

        # -------- MAIN LAYOUT
        layout_MAIN_H.addLayout(layout_V0)
        layout_MAIN_H.addLayout(layout_V1)
        self.setLayout(layout_MAIN_H)

        # ---------- CONNECTIONS --------------------------------------------------
        self.btn_LOAD_IMG.clicked.connect(self.btn_LOAD_IMG_function)
        self.slider_X.valueChanged.connect(self.slider_X_change)
        self.slider_Y.valueChanged.connect(self.slider_Y_change)
        self.slider_Z.valueChanged.connect(self.slider_Z_change)



    # ---------- VARIABLES --------------------------------------------------------
    x_max_bed = 88 # mm
    y_max_bed = 88 # mm
    z_max_bed = 96*70/175 # 70 mm - total height
    img_size  = 500 # 500x500 pixels

    x = x_max_bed/2 # default start value for x
    y = y_max_bed/2
    z = z_max_bed/2


    workspace_definition = [(-x_max_bed/2,-y_max_bed/2,0), (x_max_bed/2,-y_max_bed/2,0),\
         (-x_max_bed/2,y_max_bed/2,0), (-x_max_bed/2,-y_max_bed/2,z_max_bed)]
    workspace_definition_array = [np.array(list(item)) for item in workspace_definition]

    points = []
    points += workspace_definition_array
    vectors = [
        workspace_definition_array[1] - workspace_definition_array[0],
        workspace_definition_array[2] - workspace_definition_array[0],
        workspace_definition_array[3] - workspace_definition_array[0]
    ]

    points += [workspace_definition_array[0] + vectors[0] + vectors[1]]
    points += [workspace_definition_array[0] + vectors[0] + vectors[2]]
    points += [workspace_definition_array[0] + vectors[1] + vectors[2]]
    points += [workspace_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    flag_IMG_LOADED = 0
    XY_data = []
    SIDE_data = []

    tensor_XY = []
    tensor_SIDE = []
    print('==============')
    your_mesh = mesh.Mesh.from_file('cube.stl')
    print('zzzfzz')
    # ---------- FUNCTION DEFINITIONS ---------------------------------------------
    def btn_LOAD_IMG_function(self):
        self.flag_IMG_LOADED = 1
        #files = sorted(glob.glob ("cube_run7/rect_crop/*.jpg"))
        files = sorted(glob.glob ("dataset/rect_crop/*.jpg"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        for myFile in files:
            print(myFile)
            image = imread(myFile)
            #image = cv2.equalizeHist(image)
            image = clahe.apply(image)
            self.XY_data.append(image)
    
        print('XY_data shape:', np.array(self.XY_data).shape)
        print('XY_data[idx] shape:', np.array(self.XY_data[1]).shape)
        print('--------------------------------------')

        self.tensor_XY = np.stack([np.array(list(item)) for item in self.XY_data])
        print('tensor_XY shape:', np.array(self.tensor_XY).shape)


    def slider_X_change(self):
        #self.x = self.slider_X.value()
        self.x = int(self.slider_X.value()*self.img_size/100)
        self.lbl_INSTANT_X.setText("X= " + str(np.round(self.x*self.x_max_bed/self.img_size-self.x_max_bed/2,2)) + " mm")

    def slider_Y_change(self):
        self.y = int(self.slider_Y.value()*self.img_size/100)
        self.lbl_INSTANT_Y.setText("Y= " + str(np.round(self.y*self.y_max_bed/self.img_size-self.y_max_bed/2,2)) + " mm")

    def slider_Z_change(self):
        self.z = int(self.slider_Z.value()*self.img_size/100)
        self.lbl_INSTANT_Z.setText("Z= " + str(np.round(self.z*self.z_max_bed/self.img_size,2)) + " mm")


    def _update_canvas(self):
        if (self.flag_IMG_LOADED == 1):
            self._XY_ax.clear()
            #self._XY_ax.imshow(self.XY_data[3],cmap=cm.jet)
            self._XY_ax.imshow(self.XY_data[self.slider_Z.value()-1],cmap='gray')
            self._XY_ax.scatter((int)(self.x),(int)(self.y),color='red')
            #self._XY_ax.set_title('Captured Frame')
            self._XY_ax.axvline(x=self.x,linewidth=1,c='red')
            self._XY_ax.axhline(y=self.y,linewidth=1,c='red')
            self._XY_ax.set_ylabel('Y, px')
            self._XY_ax.set_xlabel('X, px')
            #self._XY_ax.grid()
            self._XY_ax.figure.canvas.draw()

            self._SIDE_ax.clear()
            cont_view = self.XY_data[self.slider_Z.value()-1][120:380,150:410]
            cont_view = filters.median(np.asarray((cont_view),dtype=np.uint8),selem=np.ones((13,13)))
            cont_view = cv2.Canny(cont_view,20,50)
            self._SIDE_ax.imshow(cont_view,cmap='Greys')
            self._SIDE_ax.scatter((int)(self.x)-150,(int)(self.y)-120,color='royalblue')
            self._SIDE_ax.axvline(x=self.x-150,linewidth=1,c='royalblue')
            self._SIDE_ax.axhline(y=self.y-120,linewidth=1,c='royalblue')
            self._SIDE_ax.set_ylabel('Y, px')
            self._SIDE_ax.set_xlabel('X, px')
            #self._SIDE_ax.grid()
            #self._SIDE_ax.axis("Off")
            self._SIDE_ax.figure.canvas.draw()

            self._XZ_ax.clear()
            #self._XZ_ax.imshow(self.XY_data[-1],cmap=cm.jet)
            #                                   z     y       x
            #self._XZ_ax.imshow(cv2.equalizeHist(self.tensor_XY[::-1,0:500,(int)(self.x)]),cmap='gray')
            self._XZ_ax.imshow(self.tensor_XY[::-1,0:500,(int)(self.x)],cmap='gray')
            #self._XZ_ax.scatter((int)(self.x),(int)(self.z),color='r')
            self._XZ_ax.set_ylabel('Z, px')
            self._XZ_ax.set_xlabel('X, px')
            #self._XZ_ax.set_title('XZ Slice')
            #self._XZ_ax.grid()
            self._XZ_ax.set_aspect(4)
            self._XZ_ax.figure.canvas.draw()

            self._YZ_ax.clear()
            #                                   z       y        x
            self._YZ_ax.imshow(self.tensor_XY[::-1,(int)(self.y),:],cmap='gray')
            #self._YZ_ax.scatter((int)(self.y),(int)(self.z),color='r')
            self._YZ_ax.set_ylabel('Z, px')
            self._YZ_ax.set_xlabel('Y, px')
            #self._YZ_ax.set_title('YZ Slice')
            #self._YZ_ax.grid()
            self._YZ_ax.set_aspect(4)
            self._YZ_ax.figure.canvas.draw()

            #---- workspace
            self._workspace_ax.clear()
            faces = Poly3DCollection(self.edges, linewidths=1, edgecolors='gray')
            faces.set_facecolor((0.1,0.2,1,0.03))
            self._workspace_ax.add_collection3d(faces)
            self._workspace_ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2], s=0)


            self._workspace_ax.plot([0-self.x_max_bed/2,self.x_max_bed-self.x_max_bed/2],\
                [self.y*self.y_max_bed/self.img_size-self.y_max_bed/2,self.y*self.y_max_bed/self.img_size-self.y_max_bed/2],\
                                    [self.z*self.z_max_bed/self.img_size,self.z*self.z_max_bed/self.img_size],color = 'red',linewidth=2)
            self._workspace_ax.plot([self.x*self.x_max_bed/self.img_size-self.x_max_bed/2,self.x*self.x_max_bed/self.img_size-self.x_max_bed/2],\
                                    [0-self.y_max_bed/2,self.y_max_bed-self.y_max_bed/2],\
                                    [self.z*self.z_max_bed/self.img_size,self.z*self.z_max_bed/self.img_size],color = 'limegreen',linewidth=2)
            self._workspace_ax.plot([self.x*self.x_max_bed/self.img_size-self.x_max_bed/2,self.x*self.x_max_bed/self.img_size-self.x_max_bed/2],\
                                    [self.y*self.y_max_bed/self.img_size-self.y_max_bed/2,self.y*self.y_max_bed/self.img_size-self.y_max_bed/2],\
                                    [0,self.z_max_bed],color = 'royalblue',linewidth=2)
            self._workspace_ax.scatter3D(self.x*self.x_max_bed/self.img_size-self.x_max_bed/2,\
                                         self.y*self.y_max_bed/self.img_size-self.y_max_bed/2,\
                                         self.z*self.z_max_bed/self.img_size,color = 'darkblue',s=50)
            #------------STL
            self._workspace_ax.add_collection3d(mplot3d.art3d.Poly3DCollection(self.your_mesh.vectors,\
                alpha=0.08,facecolor='blue'))
            self._workspace_ax.add_collection3d(mplot3d.art3d.Line3DCollection(self.your_mesh.vectors,\
                alpha=0.2,linewidths=1,color='black',linestyle='-'))
            #scale = self.your_mesh.points.flatten(-1)
            #self._workspace_ax.auto_scale_xyz(scale, scale, scale)

            #self._workspace_ax.set_aspect('equal')
            self._workspace_ax.set_xlim(-self.x_max_bed/2,self.x_max_bed/2)
            self._workspace_ax.set_ylim(-self.y_max_bed/2,self.y_max_bed/2)
            self._workspace_ax.set_zlim(0,70)

            self._workspace_ax.set_title('3D Printer Workspace')
            self._workspace_ax.set_ylabel('Y, mm')
            self._workspace_ax.set_xlabel('X, mm')
            self._workspace_ax.set_zlabel('Z, mm')
            self._workspace_ax.figure.canvas.draw()
        else:
            pass



# ----------- MAIN ----------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()




