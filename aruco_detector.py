#python3 aruco_detector.py --image images/example_01.png --type DICT_5X5_100

# import the necessary packages
import argparse
import imutils
import cv2
import sys
import numpy as np


class AruCoDetector:
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


    def run(self, filename, type, resize_width, vis=False):
        # load the input image from disk and resize it
        print("[INFO] loading image...")
        
        path = self.save_path + filename
        image = cv2.imread(path)
        image = imutils.resize(image, width=resize_width)
      
        # cv2.imshow('asdf', image)
        # cv2.waitKey(0)

        # load the ArUCo dictionary, grab the ArUCo parameters, and detect
        # the markers
        print("[INFO] detecting '{}' tags...".format(type))

        arucoDict = cv2.aruco.Dictionary_get(self.ARUCO_DICT[type])
        arucoParams = cv2.aruco.DetectorParameters_create()

        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
            parameters=arucoParams)
        
        centers = {}
        
        for i in range(11,16):
            centers[str(i)] = [-1, -1]
            
        # verify *at least* one ArUco marker was detected
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            cnt = 0
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))


                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                print("Center coordinate(x,y): ", cX, cY)

                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                print("[INFO] ArUco marker ID: {}".format(markerID))
                
                result = (markerID, cX, cY)
                id = str(markerID)
                
                centers[id] = [cX, cY]
                
                if vis is True:
                    # show the output image

                    cv2.imshow("Image", image)
                    # cv2.waitKey(0)
                    
                    if markerID == ids[-1]:
                        cv2.waitKey(0)
            
        
        # Validation Check
        isValid = True
        for i in range(11,16):
            id = str(i)
            isValid = centers[id] != [-1, -1]
            if isValid is False:
                print('==========Error: Marker ' + id + ' is not detected')
                
                return None   
        return centers
           
# detector = AruCoDetector('images/')

# rere = detector.run('test3.png', 'DICT_4X4_100', True)
# print(rere)