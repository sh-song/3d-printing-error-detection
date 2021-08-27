import numpy as np
import cv2
import imutils


class PerspectiveTransformer:
    def __init__(self, save_path):
        self.save_path = save_path
        

    def run(self, filename, centers, original_wh, scaling_factor, resize_width=800, vis=False):
        
        path = self.save_path + filename
        img = cv2.imread(path)
        img = imutils.resize(img, width=resize_width)

        rows, cols = img.shape[:2]
        draw = img.copy()
        pts = np.zeros((4,2), dtype=np.float32)


        topLeft = centers['11']
        topRight = centers['12']
        bottomRight = centers['13']
        bottomLeft = centers['14']

        # 변환 전 4개 좌표 
        pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])

 
        width = original_wh[0] * scaling_factor
        height = original_wh[1] * scaling_factor
        print('====================Width, Height:', width, height)
        
        # 변환 후 4개 좌표
        pts2 = np.float32([[0,0], [width-1,0], 
                            [width-1,height-1], [0,height-1]])


        # 변환 행렬 계산 
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        # 원근 변환 적용
        result = cv2.warpPerspective(img, mtrx, (width, height))
        if vis:
            cv2.imshow('scanned', result)
            cv2.waitKey(0)

        return result

