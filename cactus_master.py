#docker opencv-env2(marker dockerfile)
import numpy as np
import cv2
import imutils
from skimage import filters

from aruco_detector import AruCoDetector 
from perspective_transform import PerspectiveTransformer
from nozzle_insde_check import NozzleChecker
from edge_detector import EdgeDetector



class CactusMaster:
    def __init__(self):        
        # Set Environment
        self.save_path = 'images/'  
        self.filename = 'cac_print_2.jpg'
        self.resize_width = 1200
        self.original_wh = [154, 170] # in integer
        self.scaling_factor = 3

        # Initialize Modules
        self.detector = AruCoDetector(self.save_path)
        self.transformer = PerspectiveTransformer(self.save_path)
        self.checker = NozzleChecker()
        self.edge_detector = EdgeDetector()
        
    def preprocessing(self):
        # Detect Markers
        centers = self.detector.run(self.filename, 'DICT_4X4_100', self.resize_width, True)
        print(centers)
    
        if centers: # All markers are detected
            isNozzle = self.checker.isNozzleInside(centers)
            
            if not isNozzle: # Nozzle is outside of ROI
                print('========Execute Perspective Transform')
                topview = self.transformer.run(self.filename, centers, self.original_wh, self.scaling_factor, self.resize_width, True)
                
                edged_topview = self.edge_detector.run(topview, True)
                return edged_topview

            else:
                print('========Nozzle still inside ROI')

        return None



                
    def run(self):

        edged_topview = self.preprocessing()
        ######SYSY
        output = self.sysysysy()

        ######SYSY


if __name__ == '__main__':
    master = CactusMaster() 
    master.run()
    cv2.destroyAllWindows()