#docker opencv-env2(for_cactus dockerfile)
import numpy as np
import cv2
import imutils
from skimage import filters

from aruco_detector import AruCoDetector 
from perspective_transform import PerspectiveTransformer
from nozzle_insde_check import NozzleChecker
from edge_detector import EdgeDetector



# Detect Markers



class CactusMaster:
    def __init__(self):
        
        # Set Environment
        self.save_path = 'images/'  
        self.filename = 'cac_print_1.jpg'
        self.resize_width = 1200
        
        # Initialize Modules
        self.detector = AruCoDetector(self.save_path)
        self.transformer = PerspectiveTransformer(self.save_path)
        self.checker = NozzleChecker()
        self.edge_detector = EdgeDetector()


        isRunning = False
    def preprocessing(self):
        centers = self.detector.run(self.filename, 'DICT_4X4_100', self.resize_width, True)
        print(centers)
        
        if centers: # All markers are detected
            isNozzle = self.checker.isNozzleInside(centers)
            
            ##############DEV########3
            isNozzle = False
            ##############3
            if not isNozzle: # Nozzle is outside of ROI
                print('========Execute Perspective Transform')
                topview = self.transformer.run(self.filename, centers, self.resize_width, False)
                
                edged_topview = self.edge_detector.run(topview, True)

            else:
                print('========Nozzle still inside ROI')
                exit()
                
                
    def run(self):
        while True:
            inputtt = input('Go?')
            self.preprocessing()
                        


master = CactusMaster()
master.run()