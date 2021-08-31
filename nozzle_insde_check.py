class NozzleChecker:
    def __init__(self):
        pass
    
    def isNozzleInside(self, centers):
          
        topLeft = centers['11']
        topRight = centers['12']
        bottomRight = centers['13']
        bottomLeft = centers['14']
        nozzle = centers['15']
        
        slopes = {}
        intercepts = {}

        slopes['left'] = (topLeft[0] - bottomLeft[0]) / (topLeft[1] - bottomLeft[1] + 0.0001)
        intercepts['left'] = topLeft[0] - slopes['left'] * topLeft[1]

        slopes['right'] = (topRight[0] - bottomRight[0]) / (topRight[1] - bottomRight[1] + 0.0001)
        intercepts['right'] = topRight[0] - slopes['right'] * topRight[1]

        slopes['top'] = (topLeft[0] - topRight[0]) / (topLeft[1] - topRight[1] + 0.0001)
        intercepts['top'] = topLeft[0] - slopes['top'] * topLeft[1]

        slopes['bottom'] = (bottomLeft[0] - bottomRight[0]) / (bottomLeft[1] - bottomRight[1] + 0.0001)
        intercepts['bottom'] = bottomLeft[0] - slopes['bottom'] * bottomLeft[1]

        y = nozzle[0]
        x = nozzle[1]
        

        checks = {} # Check if nozzle is outside each line

        if slopes['left'] >= 0:
            checks['left'] = slopes['left'] * x + intercepts['left'] <= y
        else:
            checks['left'] = slopes['left'] * x + intercepts['left'] > y
 
        if slopes['right'] >= 0:
            checks['right'] = slopes['right'] * x + intercepts['right'] >= y
        else:
            checks['right'] = slopes['right'] * x + intercepts['right'] < y
        
        checks['top'] = slopes['top'] * x + intercepts['top'] <= y
    
        checks['bottom'] = slopes['bottom'] * x + intercepts['bottom'] >= y
    

        print('======================', checks)
        return checks['left'] and checks['right'] and checks['top'] and checks['bottom']