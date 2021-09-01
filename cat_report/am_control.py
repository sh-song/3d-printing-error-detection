
import numpy as np
#from stl import mesh
import sys
import cv2
import time

#--- for stl
import meshcut
from stl import mesh
import numpy.linalg as la

#--- for serial monitor
import threading
import queue
import serial

#--- for gcode parser
from pygcode import *
from pygcode import Line
from pygcode import Machine, GCodeRapidMove

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QComboBox, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtWidgets import QMainWindow, QPlainTextEdit, QCheckBox

from PyQt5.QtGui import QIcon

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from mpl_toolkits import mplot3d
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

from PIL import Image


#=======================================================================================

# GLOBAL VARIABLES
flag_TT = 0
flag_READY_TO_PRINT = 0
flag_COMMAND_TO_PRINT = 0


line_command_bank = []
cmd_to_send = ""

gcode_line_number = 0
# --------------------------------------------------------

class ThreadingTasks(QtCore.QObject):
    def __init__(self):
        super().__init__()
        print ("ThreadinTasks initialized")
        
        
    arr = [2,3,8,9]
    num = 5
    rcv = []
    #th_port = serial.Serial("/dev/ttyACM0", baudrate=9600, timeout=0)
    timer_id = 0
        
    def test_func(self, numbers):
        print ("test func activated")
        for n in numbers:
            time.sleep(1)
            print ('cube:',n*n*n)

    def portMonitor(self):
        #self.timer_id = self.startTimer(1000, timerType = QtCore.Qt.VeryCoarseTimer)
        pass
        '''global flag_READY_TO_PRINT
        while True:
            self.rcv = self.th_port.read(32).decode('utf-8')
            print(self.rcv)
            print ("READY_TO_PRINT = " + str(flag_READY_TO_PRINT))
            str_split = []
            str_split = self.rcv.split()
            for i in range(len(str_split)):
                if (str(str_split[i]) == 'ok'):
                    print('=== OK ===')
                    flag_READY_TO_PRINT = 1
                    #time.sleep(5)
                else:
                    flag_READY_TO_PRINT = 0
                    print('=== NOT OK ===')
                    
            self.rcv = []'''

    def timerEvent(self, event):
        global flag_READY_TO_PRINT
        self.rcv = self.th_port.read(255).decode('utf-8')
        print(self.rcv)
        print ("READY_TO_PRINT = " + str(flag_READY_TO_PRINT))
        #str_split = []
        str_split = self.rcv.split()
        for i in range(len(str_split)):
            if (str(str_split[i]) == 'ok'):
                print('=== OK ===')
                flag_READY_TO_PRINT = 1
                #time.sleep(5)
            else:
                flag_READY_TO_PRINT = 0
                print('=== NOT OK ===')
        #self.rcv = []


    def printingLayer(self):
        print ("Layer is printing")
        global flag_COMMAND_TO_PRINT
        global flag_READY_TO_PRINT
        global line_command_bank
        global cmd_to_send
        global gcode_line_number
        while (gcode_line_number != len(line_command_bank)):
            if (flag_READY_TO_PRINT == 1):
                flag_READY_TO_PRINT = 0
                flag_COMMAND_TO_PRINT = 0
                cmd_to_send = line_command_bank[gcode_line_number]
                self.th_port.write(('\n'+str(cmd_to_send)+'\n').encode())
                print('[{}/{}] '.format(str(gcode_line_number), \
                    str(len(line_command_bank)-1)) + str(cmd_to_send))
                gcode_line_number = gcode_line_number + 1
        '''while (len(line_command_bank) != 0):
            for j in range(gcode_line_number+1,len(line_command_bank)):
                cmd_to_send = line_command_bank[j]
                if (flag_READY_TO_PRINT == 1):
                    flag_READY_TO_PRINT = 0
                    gcode_line_number = gcode_line_number +1
                    #print ("p-->" + str(cmd_to_send))
                    self.th_port.write(('\n'+str(cmd_to_send)+'\n').encode())
                    print('[{}/{}] '.format(str(j), str(len(line_command_bank)-1)) + str(cmd_to_send))'''




class App(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'AM CONTROL APP'
        self.th_obj = ThreadingTasks();
        #self.left = 100
        #self.top = 50
        self.initUI()
 
    def initUI(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setWindowTitle(self.title)

        # ---------- COMPONENTS ---------------------------------------------------
        # -------- V0
        self.lbl_blank_space = QtWidgets.QLabel(' ')
        
        self.lbl_LOGO   = QtWidgets.QLabel('AM CONTROL')
        self.lbl_LOGO.setStyleSheet("QLabel {color: #000000}")
        self.lbl_LOGO.setFont(QtGui.QFont('Sans',22,QtGui.QFont.Bold))
        self.lbl_LOGO.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.textbox_FILENAME = QtWidgets.QLineEdit()
        self.btn_SET_FILENAME = QtWidgets.QPushButton('&Filename')

        self.btn_LOAD_STL = QtWidgets.QPushButton('&Load STL file')

        self.btn_PARSE_GCODE = QtWidgets.QPushButton('&Upload GCode')
        self.btn_PARSE_GCODE.setStyleSheet("QPushButton {background-color: #FEE101;}")

        # GCode Parser Text Field
        self.textfield_GCODE_PARSER = QPlainTextEdit(self)
        self.textfield_GCODE_PARSER.insertPlainText("textfield_GCODE_PARSER\n")
        self.textfield_GCODE_PARSER.setReadOnly(True)
        self.textfield_GCODE_PARSER.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.textfield_GCODE_PARSER.setFont(QtGui.QFont('Courier',6))

        self.lbl_GCODE_INFO_SECTION   = QtWidgets.QLabel('Total number of layers = ...\nActive layer = ...\nNumber of line-commands = ...')
        self.lbl_GCODE_INFO_SECTION.setStyleSheet("QLabel {; color: #000000;}")
        self.lbl_GCODE_INFO_SECTION.setFont(QtGui.QFont('Courier',10))

        # GCode Layer Slider
        self.lbl_GCODE_LAYER_SLIDER   = QtWidgets.QLabel('GCode Layer Slider')
        self.lbl_GCODE_LAYER_SLIDER.setStyleSheet("QLabel {background-color: #CCCCCC; color: #FFFFFF}")
        self.lbl_GCODE_LAYER_SLIDER.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))
        self.lbl_GCODE_LAYER_SLIDER.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)  

        self.slider_L = QSlider(Qt.Horizontal)
        self.slider_L.setFocusPolicy(Qt.StrongFocus)
        self.slider_L.setTickPosition(QSlider.TicksAbove)
        self.slider_L.setTickInterval(2)
        self.slider_L.setSingleStep(1)

        self.btn_VISUALIZE_LAYER = QtWidgets.QPushButton('&Draw active layer')
        self.btn_PRINT_ACTIVE_LAYER = QtWidgets.QPushButton('&Print active layer')
        self.btn_PRINT_ACTIVE_LAYER.setStyleSheet("QPushButton {background-color: #FEE101;}")


        self.btn_RUN_SERIAL_MONITOR = QtWidgets.QPushButton('&Run Serial Monitor')

        self.textfield_SERIAL_MONITOR = QPlainTextEdit(self)
        self.textfield_SERIAL_MONITOR.insertPlainText("textfield_SERIAL_MONITOR\n")
        self.textfield_SERIAL_MONITOR.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.textfield_SERIAL_MONITOR.setFont(QtGui.QFont('Courier',6))

        self.lbl_SERIAL_COMMAND_BLOCK_HEAD   = QtWidgets.QLabel('Command Block')
        self.lbl_SERIAL_COMMAND_BLOCK_HEAD.setStyleSheet("QLabel {background-color: #CCCCCC; color: #FFFFFF}")
        self.lbl_SERIAL_COMMAND_BLOCK_HEAD.setFont(QtGui.QFont('Sans',10,QtGui.QFont.Bold))
        self.lbl_SERIAL_COMMAND_BLOCK_HEAD.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        self.lbl_SERIAL_MONITOR_TEMPERATURE   = QtWidgets.QLabel('Temperature = ')
        self.lbl_SERIAL_MONITOR_TEMPERATURE.setStyleSheet("QLabel {; color: #000000;}")
        self.lbl_SERIAL_MONITOR_TEMPERATURE.setFont(QtGui.QFont('Courier',10))

        self.line_edit_COMMAND_LINE = QLineEdit(self)
        self.btn_COMMAND_SEND = QtWidgets.QPushButton('&Send Command')

        # -------- V3
        self.btn_OPEN_CAMERA = QtWidgets.QPushButton('&Open Camera')
        self.btn_OPEN_CAMERA.setStyleSheet("QPushButton {background-color: #A7A7AD;}")
        

        # ---------- LAYOUT -------------------------------------------------------
        layout_MAIN_H = QtWidgets.QHBoxLayout()
        layout_V0 = QtWidgets.QVBoxLayout()
        layout_V1 = QtWidgets.QVBoxLayout()
        layout_V2 = QtWidgets.QVBoxLayout()
        layout_V3 = QtWidgets.QVBoxLayout()
        layout_H_X = QtWidgets.QHBoxLayout()
        layout_H_Y = QtWidgets.QHBoxLayout()
        layout_H_Z = QtWidgets.QHBoxLayout()
        layout_instant_XYZ = QtWidgets.QHBoxLayout()
        layout_H_FILENAME = QtWidgets.QHBoxLayout()


        # -------- V0
        layout_V0.addWidget(self.lbl_LOGO)

        layout_H_FILENAME.addWidget(self.textbox_FILENAME)
        layout_H_FILENAME.addWidget(self.btn_SET_FILENAME)
        layout_V0.addLayout(layout_H_FILENAME)

        layout_V0.addWidget(self.btn_LOAD_STL)
        layout_V0.addWidget(self.btn_PARSE_GCODE)
        layout_V0.addWidget(self.lbl_GCODE_INFO_SECTION)
        layout_V0.addWidget(self.lbl_GCODE_LAYER_SLIDER)
        layout_V0.addWidget(self.slider_L)
        layout_V0.addWidget(self.textfield_GCODE_PARSER)
        layout_V0.addWidget(self.btn_VISUALIZE_LAYER)
        layout_V0.addWidget(self.btn_PRINT_ACTIVE_LAYER)
        layout_V0.addWidget(self.btn_RUN_SERIAL_MONITOR)
        layout_V0.addWidget(self.textfield_SERIAL_MONITOR)
        layout_V0.addWidget(self.lbl_SERIAL_COMMAND_BLOCK_HEAD)
        layout_V0.addWidget(self.line_edit_COMMAND_LINE)
        layout_V0.addWidget(self.btn_COMMAND_SEND)
        #layout_V0.addWidget(self.lbl_SERIAL_MONITOR_TEMPERATURE)
        layout_V0.addWidget(self.btn_OPEN_CAMERA)

        # -------- V3
        #layout_V3.addWidget(self.btn_OPEN_CAMERA)


        # -------- MAIN LAYOUT
        layout_MAIN_H.addLayout(layout_V0)
        layout_MAIN_H.addLayout(layout_V1)
        layout_MAIN_H.addLayout(layout_V2)
        layout_MAIN_H.addLayout(layout_V3)
        self.setLayout(layout_MAIN_H)


        # ---------- CONNECTIONS --------------------------------------------------
        self.btn_SET_FILENAME.clicked.connect(self.btn_SET_FILENAME_function)
        self.btn_LOAD_STL.clicked.connect(self.btn_LOAD_STL_function)

        self.btn_PARSE_GCODE.clicked.connect(self.btn_PARSE_GCODE_function)
        self.slider_L.valueChanged.connect(self.slider_L_change)
        self.btn_VISUALIZE_LAYER.clicked.connect(self.btn_VISUALIZE_LAYER_function)
        self.btn_PRINT_ACTIVE_LAYER.clicked.connect(self.btn_PRINT_ACTIVE_LAYER_function)

        self.btn_RUN_SERIAL_MONITOR.clicked.connect(self.btn_RUN_SERIAL_MONITOR_function)
        #self.btn_RUN_SERIAL_MONITOR.clicked.connect(self.th_obj.portMonitor)
        self.btn_COMMAND_SEND.clicked.connect(self.btn_COMMAND_SEND_function)

        self.btn_OPEN_CAMERA.clicked.connect(self.btn_OPEN_CAMERA_function)


    # ---------- VARIABLES --------------------------------------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ---- FLAGS
    flag_UPD_CANVAS = 1
    flag_STL_LOADED = 0
    flag_GCODE_PARSED = 0
    # ----

    # ---- GCode Parser
    parsed_Num_of_layers = 0
    active_layer = 0
    LAYER_THICKNESS = 0.4 # mm

    layer_bank = []
    word_bank  = []
    line_bank = []

    X_active_bank = []; Y_active_bank = []; Z_active_bank = []
    G_active_bank = []; E_active_bank = []; F_active_bank = []

    command_bank = [] # bunch of commands just for the specific layer   
    #line_command_bank = [] # bunch of commands just for the specific layer    


    # ---- GLOBAL
    XY_data = []
    SIDE_data = []

    tensor_XY = []
    tensor_SIDE = []
    stl_file = []

    filename = "default"
    #cmd_to_send = ""
    #port = serial.Serial("/dev/ttyACM0", baudrate=9600, timeout=0)
    #rcv = [] # received from port


    # ---- For OpenCV
    cMo = np.array([[ 5.96398265e-01, -2.34244280e-01, -2.12089702e+02],
            [ 2.21850690e-01,  1.00361026e+00, -2.02148236e+02],
            [ 2.28244003e-05,  7.88925038e-04,  1.00000000e+00]])


    # ---------- FUNCTION DEFINITIONS ---------------------------------------------
    def btn_SET_FILENAME_function(self):
        self.filename = self.textbox_FILENAME.text()
        #print(self.filename)
        global flag_TT
        if flag_TT == 0:
            flag_TT = 1
        else:
            flag_TT = 0
    
    
    def btn_LOAD_STL_function(self):
        self.flag_STL_LOADED = 1        
        figure = plt.figure()
        axes = mplot3d.Axes3D(figure)
        # Load the STL files and add the vectors to the plot
        your_mesh = mesh.Mesh.from_file(self.filename+'.stl')
        self.stl_file = your_mesh

        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors,alpha=0.2,facecolor='royalblue'))
        axes.add_collection3d(mplot3d.art3d.Line3DCollection(your_mesh.vectors,alpha=0.3,linewidths=1,color='black',linestyle=':'))

        verts = your_mesh.vectors.reshape(-1, 3)
        faces = np.arange(len(verts)).reshape(-1, 3)
        verts, faces = meshcut.merge_close_vertices(verts, faces)
        mesh_plane = meshcut.TriangleMesh(verts, faces)

        plane_sect = (0, 0, self.active_layer*self.LAYER_THICKNESS)
        plane_norm = (0, 0, 1)
        plane_norm /= la.norm(plane_norm)
        stl_plane = meshcut.Plane(plane_sect,plane_norm)
        P = meshcut.cross_section_mesh(mesh_plane,stl_plane)
        for i in range(np.shape(P[0])[0]):
            axes.scatter(P[0][i][0],P[0][i][1],P[0][i][2],color='red',s=5)
            axes.plot([P[0][i][0],P[0][i-1][0]],[P[0][i][1],P[0][i-1][1]],[P[0][i][2],P[0][i-1][2]],color='red',linewidth=2)
        axes.text(50,20,self.active_layer*self.LAYER_THICKNESS,s='Active layer '+str(self.active_layer),fontsize=10)

        if(self.flag_GCODE_PARSED == 1):
            for i in range(len(self.X_active_bank)):
                #ax.scatter(self.X_active_bank[i],self.Y_active_bank[i], c='r', marker='o')
                axes.plot([self.X_active_bank[i],self.X_active_bank[i-1]],
                          [self.Y_active_bank[i],self.Y_active_bank[i-1]],
                          [self.Z_active_bank[i],self.Z_active_bank[i-1]],color='b')


        # Auto scale to the mesh size
        scale = your_mesh.points.flatten(-1)
        axes.auto_scale_xyz(scale, scale, scale)
        axes.set_xlabel('X, mm')
        axes.set_ylabel('Y, mm')
        axes.set_zlabel('Z, mm')
        axes.set_title('STL Model')
        # Show the plot to the screen
        plt.show()



    def btn_PARSE_GCODE_function(self):
        with open(self.filename+'.gcode', 'r') as fh:
            for line_text in fh.readlines():
                line = Line(line_text)
                w = line.block.words
                if(np.shape(w)[0] == 0):
                    pass
                else:
                    self.word_bank.append(w)
                    self.layer_bank.append(self.parsed_Num_of_layers)
                    self.line_bank.append(line_text)
                if line.comment:
                    #self.textfield_GCODE_PARSER.insertPlainText(str(line.comment.text)+'\n')
                    if (line.comment.text[0:6] == "LAYER:"):
                        self.parsed_Num_of_layers = self.parsed_Num_of_layers + 1
                        print(self.parsed_Num_of_layers)
                        self.lbl_GCODE_INFO_SECTION.setText("Total number of layers = {}\n".format(self.parsed_Num_of_layers)
                            +"Active layer = ...\n"
                            +"Number of line-commands = ...")
        self.flag_GCODE_PARSED = 1


    def slider_L_change(self):
        global line_command_bank
        global gcode_line_number
        gcode_line_number = 0
        self.active_layer = int(self.slider_L.value()*self.parsed_Num_of_layers/100)
        #self.lbl_ACTIVE_GCODE_LAYER.setText("Active layer = " + str(self.active_layer) + '\nNumber of line-commands = '+str(len(line_command_bank)))
        self.lbl_GCODE_INFO_SECTION.setText("Total number of layers = {}\n".format(self.parsed_Num_of_layers)
                            +"Active layer = {}\n".format(str(self.active_layer))
                            +"Number of line-commands = {}".format(str(len(line_command_bank))))
        self.X_active_bank = []; self.Y_active_bank = []; self.Z_active_bank = []
        self.G_active_bank = []; self.E_active_bank = []; self.F_active_bank = []
        self.command_bank = []
        line_command_bank = []
        #print(self.active_layer)
        for i in range(len(self.layer_bank)):
            if (self.layer_bank[i] == self.active_layer):
                #print(word_bank[i])
                line_command_bank.append(self.line_bank[i])
                for j in range(len(self.word_bank[i])):
                    #print(word_bank[i][j])
                    self.command_bank.append(str(self.word_bank[i][j]))

                    if (str(self.word_bank[i][j])[:1] == 'G'):
                        self.G_active_bank.append(float(str(self.word_bank[i][j])[1:]))
                    if (str(self.word_bank[i][j])[:1] == 'X'):
                        self.X_active_bank.append(float(str(self.word_bank[i][j])[1:]))
                    if (str(self.word_bank[i][j])[:1] == 'Y'):
                        self.Y_active_bank.append(float(str(self.word_bank[i][j])[1:]))
                    if (str(self.word_bank[i][j])[:1] == 'Z'):
                        self.Z_active_bank.append(float(str(self.word_bank[i][j])[1:]))
                    if (str(self.word_bank[i][j])[:1] == 'E'):
                        self.E_active_bank.append(float(str(self.word_bank[i][j])[1:]))
                    if (str(self.word_bank[i][j])[:1] == 'F'):
                        self.F_active_bank.append(float(str(self.word_bank[i][j])[1:]))
        self.textfield_GCODE_PARSER.clear()
        for i in range(len(line_command_bank)):
            self.textfield_GCODE_PARSER.insertPlainText(line_command_bank[i])


    def btn_VISUALIZE_LAYER_function(self):
        fig = plt.figure(figsize=(6, 6),dpi=80)
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(self.X_active_bank)):
            #ax.scatter(self.X_active_bank[i],self.Y_active_bank[i], c='r', marker='o')
            ax.plot([self.X_active_bank[i],self.X_active_bank[i-1]],
                    [self.Y_active_bank[i],self.Y_active_bank[i-1]],
                    [self.Z_active_bank[i],self.Z_active_bank[i-1]],color='b')
        
        ax.set_xlabel('X, mm')
        ax.set_ylabel('Y, mm')
        ax.set_zlabel('Z, mm')
        ax.set_title('Layer '+str(self.active_layer))
        plt.show()
                        

    def btn_PRINT_ACTIVE_LAYER_function(self):
        global flag_COMMAND_TO_PRINT
        flag_COMMAND_TO_PRINT = 1
        print("****")
        t2 = threading.Thread(name="My Thread 2", target=self.th_obj.printingLayer, args=())
        t2.setDaemon(True)
        t2.start()


#-------------------------------------------------------------
    def btn_RUN_SERIAL_MONITOR_function(self):
        self.th_obj.timer_id = self.th_obj.startTimer(100, timerType = QtCore.Qt.PreciseTimer)
        #t1 = threading.Thread(name="My Thread 1", target=self.th_obj.portMonitor, args=())
        #t1.setDaemon(True)
        #t1.start()


#-------------------------------------------------------------

    def btn_COMMAND_SEND_function(self):
        to_send = self.line_edit_COMMAND_LINE.text()
        self.port.write(('\n'+str(to_send)+'\n').encode())
        print('--> Sent command: {}'.format(to_send))


#-------------------------------------------------------------
    def btn_OPEN_CAMERA_function(self):
        cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        #cap.set(cv2.CAP_PROP_FPS, 30)

        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
 
        # get frame resolution
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        print(frame_width)
        print(frame_height)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:

                pts_top_1 = np.float32([[48,122],[487,35],[134,445],[621,337]])
                pts_top_2 = np.float32([[0,0],[500,0],[0,500],[500,500]])

                #cv2.circle(frame,(95,130),5,(255,255,255),-1)
                #cv2.circle(frame,(456,57),5,(255,255,255),-1)
                #cv2.circle(frame,(175,380),5,(255,255,255),-1)
                #cv2.circle(frame,(556,310),5,(255,255,255),-1)

                cM_top = cv2.getPerspectiveTransform(pts_top_1,pts_top_2)
                top_view = cv2.warpPerspective(frame[:,:,0],cM_top,(500,500))
                top_view = cv2.cvtColor(top_view,cv2.COLOR_GRAY2RGB)
                #top_view = 255*np.ones((500,500,3), np.uint8)

                sc = 13.2 #16.5
                sh = 240  #245

                for i in range(len(self.X_active_bank)):
                    cv2.line(top_view,(int(self.X_active_bank[i]*sc+sh+10),int(self.Y_active_bank[i]*sc+sh-5)), 
                    (int(self.X_active_bank[i-1]*sc+sh+10),int(self.Y_active_bank[i-1]*sc+sh-5)),(0,0,255),1)

                pts_side_1 = np.float32([[285,408],[565,290],[329,456],[608,345]])
                pts_side_2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

                cM_side = cv2.getPerspectiveTransform(pts_side_1,pts_side_2)
                side_view = cv2.warpPerspective(frame,cM_side,(300,300))

                cv2.putText(frame,"Main CAM: Full view",(20,30),self.font,0.6,(255,255,255),1,cv2.LINE_AA) 
                #dst = cv2.warpPerspective(frame,self.cMo,(600,600))
                cv2.imshow("Frame", frame)
                cv2.moveWindow("Frame",400,50)

                cv2.imshow("Top", top_view)
                cv2.imwrite("top1.jpg",top_view)
                cv2.moveWindow("Top",1400,50)

                #cv2.imshow("Side", side_view)
                #cv2.moveWindow("Side",1400,500)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break



# ----------- MAIN ----------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()




