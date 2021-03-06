import cv2
import matplotlib.pyplot as plt

import numpy as np
from math import hypot

X_active_wall_inner = [109.618, 102.716, 102.716, 109.618, 109.218, 103.116, 103.116, 109.218, 92.501
, 88.752, 99.26, 116.116, 126.626, 122.876, 107.688, 92.154, 88.319, 99.067
, 116.309, 127.059, 123.223, 107.688, 90.204, 108.659, 126.033, 106.442, 125.335
, 98.806, 91.876, 102.635, 94.314, 107.613, 117.805, 109.697, 122.88, 109.018
, 94.988, 114.316, 124.981, 115.041, 103.361, 91.804, 111.616, 99.379, 89.032]

Y_active_wall_inner = [87.86, 87.86, 94.762, 94.762, 88.26, 88.26, 94.362, 94.362, 103.422
, 86.989, 73.81, 73.81, 86.989, 103.422, 110.736, 103.699, 86.89, 73.41
, 73.41, 86.89, 103.699, 111.18, 92.998, 73.889, 89.229, 110.047, 85.498
, 106.37, 100.327, 91.228, 80.138, 87.78, 76.057, 89.822, 103.047, 94.842
, 104.531, 73.889, 93.835, 107.106, 94.842, 83.285, 108.754, 73.889, 87.86]



# X_inner = [109.618, 102.716, 102.716, 109.618, 92.501, 88.752, 99.26, 116.116, 126.626
# , 122.876, 107.688]
# #inner - y
# Y_inner = [87.86, 87.86, 94.762, 94.762, 103.422, 86.989, 73.81, 73.81, 86.989
# , 103.422, 110.736]




 # inner - x
X_inner = [109.618,   92.501, 88.752, 99.26, 116.116, 126.626, 122.876, 107.688]
#inner - y
Y_inner = [87.86,  103.422, 86.989, 73.81, 73.81, 86.989, 103.422, 110.736]
outer = np.array((X_inner, Y_inner)).T

X_inner.append(X_inner[0])
Y_inner.append(Y_inner[0])

for first in outer:
    for second in outer:
        if first[0] != second[0]:
            delta_x = second[0] - first[0]
            delta_y = second[1] - first[1]
            dist = hypot(delta_x, delta_y)
            if (16< dist < 17):
                Xs = [first[0], second[0]]
                Ys = [first[1], second[1]]
                plt.plot(Xs, Ys)





# plt.plot(X, Y, 'g')
plt.show()    


# plt.plot(X_active_wall_inner, Y_active_wall_inner, 'b')

