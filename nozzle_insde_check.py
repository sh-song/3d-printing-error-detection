class NozzleChecker:
    def __init__(self):
        pass
    
    def isNozzleInside(self, centers):
          
        topLeft = centers['11']
        topRight = centers['12']
        bottomRight = centers['13']
        bottomLeft = centers['14']
        nozzle = centers['15']
        

        a_left = (topLeft[0] - bottomLeft[0]) / (topLeft[1] - bottomLeft[1] + 0.0001)
        b_left = topLeft[0] - a_left * topLeft[1]

        a_right = (topRight[0] - bottomRight[0]) / (topRight[1] - bottomRight[1] + 0.0001)
        b_right = topRight[0] - a_right * topRight[1]

        a_top = (topLeft[0] - topRight[0]) / (topLeft[1] - topRight[1] + 0.0001)
        b_top = topLeft[0] - a_top * topLeft[1]

        a_bottom = (bottomLeft[0] - bottomRight[0]) / (bottomLeft[1] - bottomRight[1] + 0.0001)
        b_bottom = bottomLeft[0] - a_bottom * bottomLeft[1]

        y = nozzle[0]
        x = nozzle[1]

        left_check = a_left * x + b_left < y
        right_check = a_right * x + b_right > y
        top_check = a_top * x + b_top > y
        bottom_check = a_bottom * x + b_bottom < y

        print('==========Nozzle Check:', left_check, right_check, top_check, bottom_check)
        
        return left_check and right_check and top_check and bottom_check
