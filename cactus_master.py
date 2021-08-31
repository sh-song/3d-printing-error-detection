#docker opencv-env2(marker dockerfile)
import numpy as np
import cv2
import imutils
from skimage import filters

from aruco_detector import AruCoDetector 
from perspective_transform import PerspectiveTransformer
from nozzle_insde_check import NozzleChecker
from edge_detector import EdgeDetector
from gcode_parser import Parser


class CactusMaster:
    def __init__(self):        
        # Set Environment
        #---images---
        self.save_path = 'images/'  
        self.filename = 'cac_print_2.jpg'
        # self.filename = 'outside.png'

        #---gcode---
        self.data_path = './gcode_file'
        self.file_name = 'cactus.gcode'
        self.edited_file_path_n_name = './edited_gcode/gcode_edited.gcode'
        self.slicing_times = 10 # 10 times of image processing
        self.layer_num = 2 # number of layer which is being processed
        self.layer_image_path = './layer_images_from_gcode'
        #---img_process---
        self.resize_width = 1200
        self.original_wh = [154, 170] # in integer
        self.scaling_factor = 3

        # Initialize Modules
        self.detector = AruCoDetector(self.save_path)
        self.transformer = PerspectiveTransformer(self.save_path)
        self.checker = NozzleChecker()
        self.edge_detector = EdgeDetector()
        self.parser = Parser()
        
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

    def parsing_gcode(self):

        self.parser.input_file(self.data_path,self.file_name)
        self.parser.parse()
        self.parser.slicing(self.layer_num)
        self.parser.crop()
        self.parser.edit_gcode(self.edited_file_path_n_name)
        self.parser.save(self.layer_image_path,self.layer_num)
                
    def run(self):

        edged_topview = self.preprocessing()
        # self.parsing_gcode() # save the image of layer in 'layer_images_from_gcode' directory as 'No_2_Layer.png'

if __name__ == '__main__':
    master = CactusMaster() 
    master.run()
    cv2.destroyAllWindows()