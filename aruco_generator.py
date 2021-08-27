# import the necessary packages
import numpy as np
import cv2
import sys


class AruCoGenerator:
    
    def __init__(self, SAVE_PATH):
        
        self.save_path = SAVE_PATH
    # define names of each possible ArUco tag OpenCV supports
        self.ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
        }

        # verify that the supplied ArUCo tag exists and is supported by
        # OpenCV
        # if self.ARUCO_DICT.get(args["type"], None) is None:
        #     print("[INFO] ArUCo tag of '{}' is not supported".format(
        #         args["type"]))
        #     sys.exit(0)
    

    def generate(self, type, id):
        
        # load the ArUCo dictionary
        self.arucoDict = cv2.aruco.Dictionary_get(self.ARUCO_DICT[type])
      
        # allocate memory for the output ArUCo tag and then draw the ArUCo
        # tag on the output image
        print("[INFO] generating ArUCo tag type '{}' with ID '{}'".format(type, id))
              
        tag = np.zeros((300, 300, 1), dtype="uint8")
        cv2.aruco.drawMarker(self.arucoDict, id, 300, tag, 1)


        # write the generated ArUCo tag to disk and then display it to our
        # screen
        path = self.save_path + type + '_id' + str(id) + '.png'
        print('path', path)
        cv2.imwrite(path, tag)
        cv2.imshow(type, tag)
        cv2.waitKey(1000)
        

acg = AruCoGenerator('images/')


for i in range(11, 16):
    
    acg.generate(type='DICT_4X4_100', id=i)