#docker opencv-env2(marker dockerfile)
import numpy as np
import cv2
import imutils
from skimage import filters
import threading

from aruco_detector import AruCoDetector 
from perspective_transform import PerspectiveTransformer
from nozzle_insde_check import NozzleChecker
from edge_detector import EdgeDetector
from gcode_parser import Parser

import time
# Parameters
save_path = 'images/'  
filename = 'cat_lain_28.jpg'
# filename = 'outside.png'

#---gcode---
data_path = './gcode_file'
file_name = 'cactus.gcode'
edited_file_path_n_name = './edited_gcode/gcode_edited.gcode'
slicing_times = 100 # 10 times of image processing
layer_image_path = './layer_images_from_gcode'
#---img_process---
resize_width = 1200
original_wh = [154, 170] # in integer
scaling_factor = 4




parser = Parser(data_path, \
                file_name, \
                edited_file_path_n_name, \
                slicing_times, \
                layer_image_path)




# parser.input_file()
# th_parser = threading.Thread(target=parser.start, args=())
# th_parser.start()



# Initialize Modules
detector = AruCoDetector(save_path)
transformer = PerspectiveTransformer(save_path)
checker = NozzleChecker()
edge_detector = EdgeDetector()



centers = detector.run(filename, 'DICT_4X4_100', resize_width, True)
print(centers)

if centers: # All markers are detected
    isNozzle = checker.isNozzleInside(centers)
    
    if not isNozzle: # Nozzle is outside of ROI
        print('========Execute Perspective Transform')
        topview = transformer.run(filename, centers, original_wh, scaling_factor, resize_width, True)
        
        edged_topview = edge_detector.run(topview, True)

    else:
        print('========Nozzle still inside ROI')

mask = np.zeros(edged_topview.shape)
croped = edged_topview[250:500, 0:530]

# fliped_croped = cv2.flip(croped, 0)
fliped_croped = croped

mask[250:500, 0:530] = (fliped_croped == 255)
len_y, len_x = mask.shape
cv2.imshow('mask', fliped_croped)
cv2.waitKey(0)


Xs = []
Ys = []
fliped = np.zeros(mask.shape)
for y in range(len_y):
    for x in range(len_x):
        if mask[y, x] <= 0:
            fliped[y, x] = 255
        else:
            Xs.append(x)
            Ys.append(y)
            
print(Ys)
        













# start_time = time.time()
# while parser.parsing_end_check is not True:
#     print('==========Waiting for parsing to finish for(s)', time.time() - start_time)
    
#     time.sleep(3)



# template = cv2.imread('layer_images_from_gcode/No_5_Layer.png')
# gray_template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
# (thresh, bw_template) = cv2.threshold(gray_template, 127, 255, cv2.THRESH_BINARY)

# print(bw_template.shape, edged_topview.shape)
# print(edged_topview)
# reversed_topview = 255 - edged_topview

# res = cv2.matchTemplate(reversed_topview, bw_template, cv2.TM_CCOEFF_NORMED)

# cv2.imshow('sibal', res)
# cv2.waitKey(0)

# print(edged_topview.shape, gray_template.shape, res.shape)
# print('---------------------------------------everything owari')





#while