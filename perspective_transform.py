import numpy as np
import cv2
import imutils


class PerspectiveTransformer:
    def __init__(self, save_path):
        self.save_path = save_path
        

    def run(self, filename, centers, resize_width=800, vis=False):
        
        path = self.save_path + filename
        img = cv2.imread(path)
        img = imutils.resize(img, width=resize_width)

        rows, cols = img.shape[:2]
        draw = img.copy()
        pts = np.zeros((4,2), dtype=np.float32)


        bottomRight = centers['11']     # x+y가 가장 큰 값이 좌상단 좌표
        topRight = centers['12']     # x-y가 가장 작은 것이 우상단 좌표
        topLeft = centers['13']   # x+y가 가장 값이 좌상단 좌표
        bottomLeft = centers['14']   # x-y가 가장 큰 값이 좌하단 좌표

        # 변환 전 4개 좌표 
        pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])
        print(pts1)
        # 변환 후 영상에 사용할 서류의 폭과 높이 계산 ---③ 
        w1 = abs(bottomRight[0] - bottomLeft[0])    # 상단 좌우 좌표간의 거리
        w2 = abs(topRight[0] - topLeft[0])          # 하당 좌우 좌표간의 거리
        h1 = abs(topRight[1] - bottomRight[1])      # 우측 상하 좌표간의 거리
        h2 = abs(topLeft[1] - bottomLeft[1])        # 좌측 상하 좌표간의 거리
        width = max([w1, w2])                       # 두 좌우 거리간의 최대값이 서류의 폭
        height = max([h1, h2])                      # 두 상하 거리간의 최대값이 서류의 높이

        # 변환 후 4개 좌표
        pts2 = np.float32([[0,0], [width-1,0], 
                            [width-1,height-1], [0,height-1]])

        print(pts2)

        # 변환 행렬 계산 
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        # 원근 변환 적용
        result = cv2.warpPerspective(img, mtrx, (width, height))
        if vis:
            cv2.imshow('scanned', result)
            cv2.waitKey(0)

        return result

