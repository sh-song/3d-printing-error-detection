import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from math import sin, cos, atan2, pi
# from IPython.display import display, Math, Latex, Markdown, HTML



def plot_data(data_1, data_2, label_1, label_2, markersize_1=8, markersize_2=8):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    if data_1 is not None:
        x_p, y_p = data_1
        ax.plot(x_p, y_p, color='#336699', markersize=markersize_1, marker='o', linestyle=":", label=label_1)
    if data_2 is not None:
        x_q, y_q = data_2
        ax.plot(x_q, y_q, color='orangered', markersize=markersize_2, marker='o', linestyle=":", label=label_2)
    ax.legend()
    return ax

def plot_values(values, label):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(values, label=label)
    ax.legend()
    ax.grid(True)
    plt.show()
    


# Layer 2
# gcode_Xs = [108.378, 106.672, 106.272, 104.642, 104.169, 101.772, 100.585, 97.991, 97.554, 97.171, 95.123, 94.633, 84.79, 83.723, 82.011, 74.425, 72.648, 68.794, 60.143, 58.589, 55.675, 52.781, 48.128, 46.047, 44.115, 43.513, 43.056, 42.64, 41.64, 42.671, 44.36, 47.86, 55.355, 60.117, 62.117, 68.671, 71.015, 73.029, 72.967, 72.572, 72.399, 73.994, 73.807, 73.407, 72.75, 72.984, 73.162, 72.436, 71.182, 68.584, 61.666, 59.529, 53.405, 49.191, 42.436, 41.712, 39.9, 37.17, 36.924, 37.046, 37.17, 37.866, 39.984, 42.337, 45.92, 49.786, 54.997, 58.45, 65.18, 65.687, 66.706, 66.931, 67.315, 79.265, 80.031, 82.553, 86.481, 93.914, 95.633, 96.447, 98.934,
# 100.207, 101.926, 102.154, 103.199, 103.646, 104.07, 108.796, 110.313, 112.63, 115.813, 118.75, 124.912, 130.626, 134.558, 137.782, 141.05, 148.968, 153.74, 156.427, 157.245, 158.023, 160.889, 167.885, 171.651, 177.576, 179.024, 180.042, 181.43, 181.981, 181.877, 182.009, 182.181, 182.735, 182.491, 182.195, 181.069, 179.333, 176.155, 175.162, 174.228, 173.82, 173.362, 172.459, 170.046, 165.23, 163.195, 161.091, 159.563, 156.771, 154.152, 151.765, 148.917, 145.952, 144.047, 140.489, 135.657, 131.879, 129.504, 124.815, 122.903, 118.791, 114.507, 111.688]

# gcode_Ys = [124.738, 124.059, 123.727, 119.806, 118.543, 118.509, 119.456, 122.936, 123.321, 123.506, 123.051, 123.053, 128.922, 129.252, 129.74, 132.927, 133.494, 134.158, 135.541, 135.769, 136.192, 138.128, 141.11, 141.034, 140.422, 140.067, 139.771, 139.138, 136.27, 134.474, 131.501, 130.501, 129.19, 125.916, 124.859, 121.513, 120.705, 119.78, 119.469, 119.457, 117.596, 113.046, 112.053, 109.968, 105.742, 105.028, 104., 103.132, 103.188, 103.299, 103.407, 103.294, 102.317, 103.301, 105.322, 105.643, 104.605, 102.479, 100.705, 99.587, 98.485, 97.496, 95.413, 95.174, 95.65, 95.878, 94.486, 93.436, 90.968, 90.904, 90.781, 90.701,
# 90.571, 89.353, 89.12, 89., 89.409, 90.745, 91.612, 92.344, 94.24,
# 94.813, 94.255, 92.977, 89.08, 88.016, 87.871, 85.929, 85.771, 85.766,
# 86.295, 85.463, 84.632, 84.543, 85.428, 86.04, 88.511, 92.921, 93.944,
# 95.015, 95.475, 95.965, 95.811, 97.313, 99.181, 104.305, 106.115, 108.193,
# 111.92, 117.661, 121.486, 121.985, 122.622, 125.647, 130.446, 130.611, 131.062,
# 130.555, 127.692, 123.393, 117.879, 116.448, 114.854, 112.619, 109.195, 107.346,
# 106.23, 106.796, 107.21, 107.606, 108.875, 110.249, 112.547, 114.793, 117.001,
# 120.397, 122.502, 123.522, 123.979, 124.172, 124.098, 123.945, 125.184, 125.892]

# Layer 28
gcode_Xs = [92.926, 91.344, 90.176, 86.291, 82.412, 76.866, 69.754, 64.496, 64.975,
65.301, 65.441, 65.778, 68.01, 74.702, 75.367, 73.678, 73.334, 73.223,
72.876, 70.593, 70.188, 70.036, 70.131, 70.224, 71.889, 72.514, 72.914,
72.796, 73.361, 65.67, 64.399, 58.591, 58.139, 59.66, 64.915, 72.746,
73.03, 73.313, 76.88, 78.127, 81.568, 81.744, 84.691, 86.509, 87.666,
90.09, 91.73, 93.772, 96.914, 97.432, 97.962, 99.747, 102.047, 102.134,
105.924, 108.657, 111.089, 113.052, 117.311, 117.965, 123.324, 125.761, 128.201,
128.515, 129.384, 133.543, 135.379, 138.68, 142.059, 146.864, 150.389, 151.018,
152.618, 155.444, 157.773, 157.951, 158.164, 163.164, 163.195, 174.687, 177.201,
179.119, 180.165, 181.073, 181.935, 181.907, 182.084, 181.79, 179.579, 177.243,
176.762, 176.677, 174.796, 172.65, 171.555, 170.075, 165.955, 163.023, 160.148,
156.539, 155.877, 153.885, 151.824, 150.77, 146.631, 144.4, 141.863, 140.645,
135.659, 132.72, 132.072, 129.538, 127.274, 126.593, 120.78, 120.239, 115.997,
114.514, 112.58, 109.225, 105.53, 104.524, 103.165, 101.701, 99.692, 98.481,
95.87, 93.329]

gcode_Ys = [125.851, 125.856, 125.781, 126.98, 128.087, 128.658, 131.443, 130.712, 128.092,
127.759, 127.676, 127.56, 126.533, 124.058, 122.898, 117.564, 117.289, 117.182,
116.581, 112.668, 110.242, 105.278, 104.986, 104.453, 101.681, 100.625, 100.155,
99.818, 97.704, 98.592, 98.66, 98.361, 98.284, 96.503, 93.798, 92.326,
92.291, 92.122, 90.899, 90.771, 89.723, 89.671, 89.53, 89.101, 88.638,
88.914, 89.39, 90.42, 93.247, 93.557, 93.464, 93.959, 91.387, 91.287,
88.018, 87.333, 87.307, 88.015, 87.789, 87.628, 83.767, 83.274, 82.694,
82.543, 82.542, 82.364, 82.711, 83.442, 85.119, 88.404, 89.87, 90.394,
91.775, 93.774, 94.445, 94.576, 94.609, 95.135, 95.151, 101.712, 104.553,
106.95, 109.089, 113.04, 126.393, 127.392, 128.055, 128.271, 128.991, 127.689,
127.354, 126.884, 117.159, 113.116, 111.511, 110.492, 107.218, 106.695, 107.043,
108.621, 109.43, 111.216, 113.053, 114.599, 118.676, 120.322, 122.444, 123.549,
124.582, 125.223, 125.012, 124.939, 124.969, 124.851, 121.963, 121.942, 122.697,
123.732, 124.101, 124.047, 121.493, 121.242, 119.436, 119.186, 120.787, 122.411,
124.713, 125.941]

camera_Xs =[310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 303, 304, 305, 306, 307, 308, 309, 337, 338, 339, 340, 341, 297, 298, 299, 300, 301, 302, 342, 343, 344, 345, 346, 293, 294, 295, 296, 347, 348, 349, 350, 290, 291, 292, 351, 352, 353, 354, 355, 356, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 285, 286, 287, 288, 289, 357, 358, 359, 257, 258, 259, 260, 261, 284, 359, 360, 361, 254, 255, 256, 299, 361, 362, 252, 253, 300, 301, 302, 362, 363, 249, 250, 251, 302, 303, 363, 364, 246, 247, 248, 303, 304, 364, 365, 366, 195, 244, 245, 246, 304, 305, 325, 366, 367, 195, 196, 243, 244, 305, 325, 326, 367, 368, 196, 242, 243, 305, 326, 327, 368, 369, 370, 196, 197, 241, 242, 306, 327, 328, 370, 371, 241, 306, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 371, 372, 373, 193, 195, 198, 241, 306, 307, 347, 348, 349, 350, 351, 352, 373, 374, 192, 193, 194, 196, 198, 199, 240, 306, 353, 354, 355, 356, 374, 375, 376, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 197, 199, 200, 240, 306, 357, 358, 359, 376, 377, 378, 143, 157, 158, 159, 160, 161, 200, 201, 202, 240, 306, 360, 361, 362, 378, 379, 380, 139, 140, 141, 142, 144, 145, 146, 148, 149, 150, 151, 152, 153, 154, 155, 156, 202, 203, 239, 306, 362, 363, 364, 380, 381, 136, 137, 138, 147, 203, 204, 205, 206, 239, 305, 364, 365, 366, 381, 382, 383, 130, 131, 132, 133, 134, 135, 136, 207, 208, 209, 239, 305, 366, 367, 368, 383, 384, 385, 121, 122, 123, 124, 125, 126, 127, 128, 129, 210, 211, 239, 304, 368, 369, 370, 385, 386, 387, 116, 117, 118, 119, 120, 211, 212, 213, 238, 303, 370, 371, 372, 387, 388, 389, 112, 113, 114, 115, 182, 213, 214, 238, 302, 372, 373, 389, 390, 391, 109, 110, 111, 182, 183, 214, 215, 216, 238, 373, 374, 375, 391, 392, 393, 106, 107, 108, 184, 216, 217, 238, 375, 376, 393, 394, 395, 103, 104, 105, 106, 184, 185, 217, 218, 237, 376, 377, 395, 396, 397, 100, 101, 102, 185, 218, 219, 237, 377, 378, 379, 397, 398, 399, 98, 99, 185, 219, 220, 221, 237, 379, 380, 400, 401, 402, 94, 95, 96, 97, 185, 221, 222, 237, 380, 381, 382, 403, 404, 405, 406, 92, 93, 94, 186, 222, 223, 236, 382, 383, 407, 408, 409, 410, 90, 91, 92, 186, 223, 224, 225, 235, 236, 383, 384, 385, 411, 412, 413, 414, 88, 89, 186, 225, 226, 227, 234, 235, 385, 386, 387, 415, 416, 85, 86, 87, 186, 227, 228, 232, 233, 234, 387, 388, 417, 418, 419, 82, 83, 84, 186, 187, 229, 230, 231, 232, 236, 389, 390, 420, 421, 422, 78, 79, 80, 81, 187, 188, 189, 236, 390, 391, 392, 393, 422, 423, 424, 75, 76, 77, 189, 190, 235, 393, 394, 395, 424, 425, 426, 72, 73, 74, 190, 191, 192, 235, 396, 397, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 33, 34, 35, 36, 37, 38, 39, 68, 69, 70, 71, 193, 194, 195, 196, 235, 398, 399, 438, 439, 440, 441, 442, 28, 29, 30, 31, 32, 40, 41, 42, 43, 44, 45, 46, 47, 64, 65, 66, 67, 196, 197, 198, 235, 399, 400, 443, 444, 445, 446, 447, 26, 27, 28, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 198, 199, 200, 201, 202, 235, 400, 401, 448, 449, 450, 451, 452, 25, 26, 203, 204, 234, 401, 402, 403, 453, 454, 455, 456, 24, 25, 204, 205, 234, 387, 388, 403, 404, 457, 458, 459, 23, 24, 205, 206, 207, 233, 387, 404, 405, 460, 461, 462, 22, 23, 207, 208, 233, 386, 405, 406, 462, 463, 464, 21, 22, 208, 209, 233, 384, 385, 406, 407, 408, 464, 465, 466, 20, 21, 209, 210, 232, 383, 384, 408, 409, 466, 467, 468, 19, 20, 210, 211, 231, 232, 383, 409, 410, 468, 469, 19, 211, 212, 213, 230, 231, 410, 411, 469, 470, 471, 18, 19, 213, 214, 229, 230, 411, 412, 471, 472, 473, 18, 214, 215, 216, 228, 229, 412, 413, 414, 473, 474, 17, 216, 217, 218, 226, 227, 228, 414, 415, 416, 474, 475, 17, 218, 219, 220, 221, 222, 223, 224, 225, 397, 398, 416, 417, 475, 476, 477, 17, 396, 397, 477, 478, 17, 395, 396, 472, 473, 478, 479, 16, 394, 395, 473, 474, 479, 480, 16, 393, 394, 474, 475, 480, 481, 16, 393, 475, 476, 481, 482, 16, 477, 482, 483, 16, 478, 483, 484, 16, 479, 484, 485, 16, 485, 486, 487, 16, 487, 488, 17, 488, 489, 17, 489, 490, 17, 490, 491, 17, 491, 492, 18, 492, 493, 18, 19, 493, 494, 19, 20, 494, 495, 20, 21, 495, 21, 22, 495, 496, 22, 23, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 496, 497, 23, 24, 76, 77, 78, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 127, 128, 497, 24, 25, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 128, 129, 498, 25, 26, 56, 57, 58, 59, 60, 61, 129, 130, 498, 499, 26, 27, 50, 51, 52, 53, 54, 55, 130, 499, 27, 28, 45, 46, 47, 48, 49, 131, 499, 500, 28, 29, 30, 41, 42, 43, 44, 130, 479, 500, 30, 31, 32, 37, 38, 39, 40, 130, 479, 480, 481, 500, 501, 33, 34, 35, 36, 130, 446, 481, 482, 501, 129, 445, 446, 482, 483, 484, 501, 129, 348, 349, 444, 445, 484, 485, 502, 129, 347, 348, 443, 444, 485, 486, 502, 128, 346, 347, 442, 443, 486, 503, 128, 345, 346, 441, 442, 487, 503, 127, 345, 441, 487, 488, 503, 127, 488, 489, 504, 127, 489, 490, 504, 126, 490, 491, 505, 126, 491, 492, 505, 125, 492, 505, 125, 472, 473, 493, 506, 125, 471, 472, 473, 494, 506, 125, 470, 471, 494, 495, 506, 124, 469, 470, 495, 496, 506, 124, 469, 496, 507, 124, 497, 507, 124, 497, 498, 507, 124, 498, 507, 124, 438, 439, 440, 441, 442, 443, 444, 445, 446, 498, 499, 507, 124, 434, 435, 436, 437, 447, 448, 449, 450, 499, 500, 508, 124, 430, 431, 432, 433, 451, 452, 453, 500, 508, 124, 428, 429, 430, 453, 454, 455, 500, 508, 125, 426, 427, 428, 455, 456, 457, 500, 508, 125, 424, 425, 426, 457, 458, 501, 508, 125, 423, 424, 458, 459, 501, 508, 125, 422, 423, 459, 460, 461, 501, 508, 125, 420, 421, 422, 461, 462, 502, 508, 125, 419, 420, 462, 463, 502, 508, 125, 418, 419, 463, 464, 502, 508, 125, 417, 418, 464, 465, 466, 503, 508, 125, 416, 417, 466, 467, 503, 508, 125, 415, 416, 467, 468, 469, 503, 508, 125, 413, 414, 415, 469, 470, 503, 508, 125, 412, 413, 470, 471, 503, 508, 125, 411, 412, 413, 471, 504, 508, 125, 410, 411, 471, 472, 504, 508, 126, 409, 410, 472, 473, 504, 508, 126, 408, 409, 473, 474, 504, 508, 126, 407, 408, 474, 475, 504, 508, 126, 406, 407, 475, 504, 508, 127, 233, 234, 406, 475, 476, 504, 508, 127, 232, 233, 405, 406, 476, 504, 508, 127, 231, 232, 404, 405, 476, 477, 504, 508, 126, 230, 231, 403, 404, 477, 504, 508, 126, 229, 230, 403, 477, 504, 508, 125, 126, 402, 478, 504, 508, 124, 125, 401, 402, 478, 504, 508, 122, 123, 124, 400, 401, 478, 504, 505, 508, 120, 121, 122, 399, 400, 479, 505, 509, 119, 120, 398, 399, 479, 505, 509, 117, 118, 119, 397, 398, 479, 506, 509, 115, 116, 117, 396, 397, 479, 506, 509, 113, 114, 115, 396, 480, 506, 510, 111, 112, 113, 395, 396, 480, 506, 510, 109, 110, 111, 394, 395, 480, 506, 510, 107, 108, 109, 393, 394, 480, 506, 510, 105, 106, 107, 392, 393, 481, 505, 510, 103, 104, 105, 391, 392, 481, 505, 511, 102, 103, 390, 391, 481, 504, 511, 100, 101, 102, 311, 330, 389, 390, 481, 503, 504, 511, 98, 99, 100, 310, 311, 329, 330, 387, 388, 389, 482, 503, 511, 96, 97, 98, 291, 309, 310, 327, 328, 329, 386, 387, 482, 502, 503, 511, 95, 96, 289, 290, 291, 308, 309, 326, 327, 385, 386, 482, 502, 511, 93, 94, 95, 288, 289, 307, 308, 325, 326, 383, 384, 385, 482, 511, 91, 92, 93, 288, 306, 307, 382, 383, 482, 511, 90, 91, 232, 233, 234, 235, 236, 237, 381, 382, 483, 506, 511, 89, 90, 231, 232, 238, 239, 380, 381, 483, 506, 511, 87, 88, 89, 229, 230, 231, 239, 240, 241, 378, 379, 380, 483, 507, 511, 86, 87, 228, 229, 241, 242, 377, 378, 483, 507, 511, 84, 85, 86, 227, 228, 242, 243, 244, 376, 377, 483, 507, 511, 82, 83, 84, 226, 227, 244, 245, 246, 375, 376, 484, 507, 511, 81, 82, 225, 226, 247, 248, 374, 375, 484, 507, 511, 77, 78, 79, 80, 224, 225, 249, 250, 251, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 373, 374, 484, 507, 511, 72, 73, 74, 75, 76, 223, 224, 251, 252, 253, 286, 287, 288, 289, 301, 302, 303, 304, 372, 373, 484, 507, 511, 67, 68, 69, 70, 71, 222, 223, 253, 254, 283, 284, 285, 304, 305, 306, 371, 372, 484, 507, 511, 61, 62, 63, 64, 65, 66, 221, 222, 254, 255, 256, 280, 281, 282, 283, 306, 307, 308, 370, 371, 484, 508, 511, 57, 58, 59, 60, 220, 221, 256, 257, 278, 279, 280, 308, 309, 310, 368, 369, 370, 485, 508, 511, 55, 56, 219, 220, 257, 258, 259, 272, 273, 274, 275, 276, 277, 310, 311, 312, 366, 367, 368, 485, 508, 511, 51, 52, 53, 54, 218, 219, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 312, 313, 314, 362, 363, 364, 365, 366, 485, 508, 512, 49, 50, 217, 218, 314, 315, 316, 358, 359, 360, 361, 485, 508, 512, 46, 47, 48, 216, 217, 316, 317, 318, 353, 354, 355, 356, 357, 485, 486, 508, 512, 45, 46, 214, 215, 216, 318, 319, 320, 349, 350, 351, 352, 486, 508, 512, 44, 45, 213, 214, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 344, 345, 346, 347, 348, 487, 508, 512, 43, 44, 211, 212, 213, 335, 336, 337, 338, 339, 340, 341, 342, 343, 487, 508, 512, 42, 43, 210, 211, 487, 488, 508, 512, 41, 42, 208, 209, 210, 488, 508, 512, 41, 193, 194, 195, 196, 206, 207, 488, 489, 508, 511, 40, 41, 188, 189, 190, 191, 192, 197, 198, 199, 200, 201, 202, 203, 204, 205, 489, 490, 510, 511, 40, 185, 186, 187, 490, 491, 509, 510, 39, 40, 182, 183, 184, 491, 492, 508, 509, 39, 179, 180, 181, 492, 493, 507, 508, 38, 39, 175, 176, 177, 178, 493, 494, 495, 496, 505, 506, 507, 37, 38, 169, 170, 171, 172, 173, 174, 496, 497, 498, 502, 503, 504, 505, 37, 161, 162, 163, 164, 165, 166, 167, 168, 499, 500, 501, 36, 37, 154, 155, 156, 157, 158, 159, 160, 36, 149, 150, 151, 152, 153, 36, 145, 146, 147, 148, 35, 143, 144, 35, 140, 141, 142, 35, 138, 139, 35, 134, 135, 136, 137, 35, 132, 133, 134, 35, 130, 131, 35, 128, 129, 130, 36, 125, 126, 127, 36, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 36, 37, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 37, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 37, 38, 76, 77, 78, 79, 38, 39, 73, 74, 75, 39, 40, 71, 72, 73, 40, 41, 69, 70, 71, 41, 42, 68, 69, 42, 43, 66, 67, 68, 43, 44, 65, 66, 44, 45, 63, 64, 65, 45, 46, 47, 61, 62, 63, 47, 48, 49, 60, 61, 49, 50, 58, 59, 60, 50, 51, 52, 53, 56, 57, 58, 54, 55]

camera_Ys = [287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 287, 288, 288, 288, 288, 288, 288, 288, 288, 288, 288, 288, 288, 289, 289, 289, 289, 289, 289, 289, 289, 289, 289, 289, 290, 290, 290, 290, 290, 290, 290, 290, 291, 291, 291, 291, 291, 291, 291, 291, 291, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 292, 293, 293, 293, 293, 293, 293, 293, 293, 293, 294, 294, 294, 294, 294, 294, 295, 295, 295, 295, 295, 295, 295, 296, 296, 296, 296, 296, 296, 296, 297, 297, 297, 297, 297, 297, 297, 297, 298, 298, 298, 298, 298, 298, 298, 298, 298, 299, 299, 299, 299, 299, 299, 299, 299, 299, 300, 300, 300, 300, 300, 300, 300, 300, 300, 301, 301, 301, 301, 301, 301, 301, 301, 301, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 302, 303, 303, 303, 303, 303, 303, 303, 303, 303, 303, 303, 303, 303, 303, 304, 304, 304, 304, 304, 304, 304, 304, 304, 304, 304, 304, 304, 304, 304, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 306, 306, 306, 306, 306, 306, 306, 306, 306, 306, 306, 306, 306, 306, 306, 306, 306, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 307, 308, 308, 308, 308, 308, 308, 308, 308, 308, 308, 308, 308, 308, 308, 308, 308, 309, 309, 309, 309, 309, 309, 309, 309, 309, 309, 309, 309, 309, 309, 309, 309, 309, 309, 310, 310, 310, 310, 310, 310, 310, 310, 310, 310, 310, 310, 310, 310, 310, 310, 310, 310, 310, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 311, 312, 312, 312, 312, 312, 312, 312, 312, 312, 312, 312, 312, 312, 312, 313, 313, 313, 313, 313, 313, 313, 313, 313, 313, 313, 313, 313, 313, 313, 314, 314, 314, 314, 314, 314, 314, 314, 314, 314, 314, 314, 315, 315, 315, 315, 315, 315, 315, 315, 315, 315, 315, 315, 315, 315, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 317, 317, 317, 317, 317, 317, 317, 317, 317, 317, 317, 317, 318, 318, 318, 318, 318, 318, 318, 318, 318, 318, 318, 318, 318, 318, 318, 319, 319, 319, 319, 319, 319, 319, 319, 319, 319, 319, 319, 319, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 320, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 321, 322, 322, 322, 322, 322, 322, 322, 322, 322, 322, 322, 322, 322, 322, 323, 323, 323, 323, 323, 323, 323, 323, 323, 323, 323, 323, 323, 323, 323, 324, 324, 324, 324, 324, 324, 324, 324, 324, 324, 324, 324, 324, 324, 324, 325, 325, 325, 325, 325, 325, 325, 325, 325, 325, 325, 325, 326, 326, 326, 326, 326, 326, 326, 326, 326, 326, 326, 326, 326, 326, 326, 326, 326, 326, 326, 326, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 327, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 328, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 329, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 330, 331, 331, 331, 331, 331, 331, 331, 331, 331, 331, 331, 331, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 333, 333, 333, 333, 333, 333, 333, 333, 333, 333, 333, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 334, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 335, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 336, 337, 337, 337, 337, 337, 337, 337, 337, 337, 337, 337, 338, 338, 338, 338, 338, 338, 338, 338, 338, 338, 338, 339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 339, 340, 340, 340, 340, 340, 340, 340, 340, 340, 340, 340, 340, 341, 341, 341, 341, 341, 341, 341, 341, 341, 341, 341, 341, 341, 341, 341, 341, 342, 342, 342, 342, 342, 343, 343, 343, 343, 343, 343, 343, 344, 344, 344, 344, 344, 344, 344, 345, 345, 345, 345, 345, 345, 345, 346, 346, 346, 346, 346, 346, 347, 347, 347, 347, 348, 348, 348, 348, 349, 349, 349, 349, 350, 350, 350, 350, 351, 351, 351, 352, 352, 352, 353, 353, 353, 354, 354, 354, 355, 355, 355, 356, 356, 356, 357, 357, 357, 357, 358, 358, 358, 358, 359, 359, 359, 360, 360, 360, 360, 361, 361, 361, 361, 361, 361, 361, 361, 361, 361, 361, 361, 361, 361, 361, 361, 362, 362, 362, 362, 362, 362, 362, 362, 362, 362, 362, 362, 362, 362, 362, 362, 362, 362, 362, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 363, 364, 364, 364, 364, 364, 364, 364, 364, 364, 364, 364, 364, 365, 365, 365, 365, 365, 365, 365, 365, 365, 365, 366, 366, 366, 366, 366, 366, 366, 366, 366, 366, 367, 367, 367, 367, 367, 367, 367, 367, 367, 367, 368, 368, 368, 368, 368, 368, 368, 368, 368, 368, 368, 368, 368, 369, 369, 369, 369, 369, 369, 369, 369, 369, 370, 370, 370, 370, 370, 370, 370, 371, 371, 371, 371, 371, 371, 371, 371, 372, 372, 372, 372, 372, 372, 372, 372, 373, 373, 373, 373, 373, 373, 373, 374, 374, 374, 374, 374, 374, 374, 375, 375, 375, 375, 375, 375, 376, 376, 376, 376, 377, 377, 377, 377, 378, 378, 378, 378, 379, 379, 379, 379, 380, 380, 380, 381, 381, 381, 381, 381, 382, 382, 382, 382, 382, 382, 383, 383, 383, 383, 383, 383, 384, 384, 384, 384, 384, 384, 385, 385, 385, 385, 386, 386, 386, 387, 387, 387, 387, 388, 388, 388, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 389, 390, 390, 390, 390, 390, 390, 390, 390, 390, 390, 390, 390, 391, 391, 391, 391, 391, 391, 391, 391, 391, 391, 392, 392, 392, 392, 392, 392, 392, 392, 392, 393, 393, 393, 393, 393, 393, 393, 393, 393, 394, 394, 394, 394, 394, 394, 394, 394, 395, 395, 395, 395, 395, 395, 395, 396, 396, 396, 396, 396, 396, 396, 396, 397, 397, 397, 397, 397, 397, 397, 397, 398, 398, 398, 398, 398, 398, 398, 399, 399, 399, 399, 399, 399, 399, 400, 400, 400, 400, 400, 400, 400, 400, 401, 401, 401, 401, 401, 401, 401, 402, 402, 402, 402, 402, 402, 402, 402, 403, 403, 403, 403, 403, 403, 403, 403, 404, 404, 404, 404, 404, 404, 404, 405, 405, 405, 405, 405, 405, 405, 406, 406, 406, 406, 406, 406, 406, 407, 407, 407, 407, 407, 407, 407, 408, 408, 408, 408, 408, 408, 408, 409, 409, 409, 409, 409, 409, 409, 410, 410, 410, 410, 410, 410, 411, 411, 411, 411, 411, 411, 411, 411, 412, 412, 412, 412, 412, 412, 412, 412, 413, 413, 413, 413, 413, 413, 413, 413, 413, 414, 414, 414, 414, 414, 414, 414, 414, 415, 415, 415, 415, 415, 415, 415, 416, 416, 416, 416, 416, 416, 417, 417, 417, 417, 417, 417, 417, 418, 418, 418, 418, 418, 418, 418, 418, 418, 419, 419, 419, 419, 419, 419, 419, 419, 420, 420, 420, 420, 420, 420, 420, 421, 421, 421, 421, 421, 421, 421, 421, 422, 422, 422, 422, 422, 422, 422, 422, 423, 423, 423, 423, 423, 423, 423, 424, 424, 424, 424, 424, 424, 424, 424, 425, 425, 425, 425, 425, 425, 425, 425, 426, 426, 426, 426, 426, 426, 426, 426, 427, 427, 427, 427, 427, 427, 427, 427, 428, 428, 428, 428, 428, 428, 428, 428, 429, 429, 429, 429, 429, 429, 429, 430, 430, 430, 430, 430, 430, 430, 430, 430, 430, 430, 431, 431, 431, 431, 431, 431, 431, 431, 431, 431, 431, 431, 431, 432, 432, 432, 432, 432, 432, 432, 432, 432, 432, 432, 432, 432, 432, 432, 433, 433, 433, 433, 433, 433, 433, 433, 433, 433, 433, 433, 433, 433, 434, 434, 434, 434, 434, 434, 434, 434, 434, 434, 434, 434, 434, 434, 435, 435, 435, 435, 435, 435, 435, 435, 435, 435, 436, 436, 436, 436, 436, 436, 436, 436, 436, 436, 436, 436, 436, 437, 437, 437, 437, 437, 437, 437, 437, 437, 437, 437, 438, 438, 438, 438, 438, 438, 438, 438, 438, 438, 438, 438, 438, 438, 438, 439, 439, 439, 439, 439, 439, 439, 439, 439, 439, 439, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 440, 441, 441, 441, 441, 441, 441, 441, 441, 441, 441, 441, 441, 441, 442, 442, 442, 442, 442, 442, 442, 442, 442, 442, 442, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 443, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 444, 445, 445, 445, 445, 445, 445, 445, 445, 445, 445, 445, 445, 445, 445, 445, 445, 445, 445, 445, 445, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 446, 447, 447, 447, 447, 447, 447, 447, 447, 447, 447, 447, 447, 447, 447, 447, 447, 447, 447, 447, 447, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 448, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 451, 451, 451, 451, 451, 451, 451, 451, 451, 451, 451, 451, 451, 451, 451, 451, 451, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 453, 454, 454, 454, 454, 454, 454, 454, 454, 454, 454, 454, 454, 454, 454, 454, 454, 454, 455, 455, 455, 455, 455, 455, 455, 455, 456, 456, 456, 456, 456, 456, 456, 456, 457, 457, 457, 457, 457, 457, 457, 457, 457, 457, 457, 458, 458, 458, 458, 458, 458, 458, 458, 458, 458, 458, 458, 458, 458, 458, 458, 458, 458, 458, 458, 459, 459, 459, 459, 459, 459, 459, 459, 460, 460, 460, 460, 460, 460, 460, 460, 460, 461, 461, 461, 461, 461, 461, 461, 461, 462, 462, 462, 462, 462, 462, 462, 462, 462, 462, 462, 462, 462, 463, 463, 463, 463, 463, 463, 463, 463, 463, 463, 463, 463, 463, 463, 463, 464, 464, 464, 464, 464, 464, 464, 464, 464, 464, 464, 464, 465, 465, 465, 465, 465, 465, 465, 465, 465, 466, 466, 466, 466, 466, 466, 467, 467, 467, 467, 467, 468, 468, 468, 469, 469, 469, 469, 470, 470, 470, 471, 471, 471, 471, 471, 472, 472, 472, 472, 473, 473, 473, 474, 474, 474, 474, 475, 475, 475, 475, 476, 476, 476, 476, 476, 476, 476, 476, 476, 476, 476, 476, 476, 476, 477, 477, 477, 477, 477, 477, 477, 477, 477, 477, 477, 477, 477, 477, 477, 477, 477, 477, 477, 477, 478, 478, 478, 478, 478, 478, 478, 478, 478, 478, 478, 478, 478, 478, 478, 479, 479, 479, 479, 479, 479, 480, 480, 480, 480, 480, 481, 481, 481, 481, 481, 482, 482, 482, 482, 482, 483, 483, 483, 483, 484, 484, 484, 484, 484, 485, 485, 485, 485, 486, 486, 486, 486, 486, 487, 487, 487, 487, 487, 487, 488, 488, 488, 488, 488, 489, 489, 489, 489, 489, 490, 490, 490, 490, 490, 490, 490, 491, 491]





gcode_num_points = len(gcode_Xs)
print('gcode num points', gcode_num_points)
true_data = np.zeros((2, gcode_num_points))
true_data[0, :] = gcode_Xs #x?
true_data[1, :] = gcode_Ys

gcode_max_x = max(gcode_Xs)
gcode_min_x = min(gcode_Xs)
gcode_max_y = max(gcode_Ys)
gcode_min_y = min(gcode_Ys)






camera_num_points = len(camera_Xs)
sampled_Xs = []
sampled_Ys = []
factor = camera_num_points // gcode_num_points 

# sampled_num_points = 

print('------------factor', factor)
for i in range(gcode_num_points):
    sampled_Xs.append(camera_Xs[factor*i])
    sampled_Ys.append(camera_Ys[factor*i])

num_points = len(sampled_Xs)
print('============================',camera_num_points, num_points, gcode_num_points)
camera_max_x = max(sampled_Xs)
camera_min_x = min(sampled_Xs)

scaling_factor = (gcode_max_x - gcode_min_x) / (camera_max_x - camera_min_x)




moved_data = np.zeros((2, num_points))
moved_data[0, :] = sampled_Xs #x?
moved_data[1, :] = sampled_Ys
moved_data *= scaling_factor
# moved_data = R_true.dot(true_data) + t_true

# Assign to variables we use in formulas.
Q = true_data # Gcode Data
P = moved_data # Camera Data

plot_data(moved_data, true_data, "P: Camera data", "Q: Gcode data")
plt.show()


def get_correspondence_indices(P, Q):
    """For each point in P find closest one in Q."""
    p_size = P.shape[1]
    q_size = Q.shape[1]
    correspondences = []
    for i in range(p_size):
        p_point = P[:, i]
        min_dist = sys.maxsize
        chosen_idx = -1
        for j in range(q_size):
            q_point = Q[:, j]
            dist = np.linalg.norm(q_point - p_point)
            if dist < min_dist:
                min_dist = dist
                chosen_idx = j
        correspondences.append((i, chosen_idx))
    return correspondences

def draw_correspondeces(P, Q, correspondences, ax):
    label_added = False
    for i, j in correspondences:
        x = [P[0, i], Q[0, j]]
        y = [P[1, i], Q[1, j]]
        if not label_added:
            ax.plot(x, y, color='grey', label='correpondences')
            label_added = True
        else:
            ax.plot(x, y, color='grey')
    ax.legend()
    
    
    

correspondences = get_correspondence_indices(P, Q)
ax = plot_data(P, Q, "Camera data", "Gcode data")
draw_correspondeces(P, Q, correspondences, ax)
plt.show()


def dR(theta):
    return np.array([[-sin(theta), -cos(theta)],
                     [cos(theta),  -sin(theta)]])

def R(theta):
    return np.array([[cos(theta), -sin(theta)],
                     [sin(theta),  cos(theta)]])
    
    
def jacobian(x, p_point):
    theta = x[2]
    J = np.zeros((2, 3))
    J[0:2, 0:2] = np.identity(2)
    J[0:2, [2]] = dR(0).dot(p_point)
    return J

def error(x, p_point, q_point):
    rotation = R(x[2])
    translation = x[0:2]
    prediction = rotation.dot(p_point) + translation
    return prediction - q_point


def prepare_system(x, P, Q, correspondences, kernel=lambda distance: 1.0):
    H = np.zeros((3, 3))
    g = np.zeros((3, 1))
    chi = 0
    for i, j in correspondences:
        p_point = P[:, [i]]
        q_point = Q[:, [j]]
        e = error(x, p_point, q_point)
        weight = kernel(e) # Please ignore this weight until you reach the end of the notebook.
        J = jacobian(x, p_point)
        H += weight * J.T.dot(J)
        g += weight * J.T.dot(e)
        chi += e.T * e
    return H, g, chi

def icp_least_squares(P, Q, iterations=30, kernel=lambda distance: 1.0):
    x = np.zeros((3, 1))
    chi_values = []
    x_values = [x.copy()]  # Initial value for transformation.
    P_values = [P.copy()]
    P_copy = P.copy()
    corresp_values = []
    for i in range(iterations):
        rot = R(x[2])
        t = x[0:2]
        correspondences = get_correspondence_indices(P_copy, Q)
        corresp_values.append(correspondences)
        H, g, chi = prepare_system(x, P, Q, correspondences, kernel)
        dx = np.linalg.lstsq(H, -g, rcond=None)[0]
        x += dx
        x[2] = atan2(sin(x[2]), cos(x[2])) # normalize angle
        chi_values.append(chi.item(0))
        x_values.append(x.copy())
        rot = R(x[2])
        t = x[0:2]
        P_copy = rot.dot(P.copy()) + t
        P_values.append(P_copy)
    corresp_values.append(corresp_values[-1])
    return P_values, chi_values, corresp_values

P_values, chi_values, corresp_values = icp_least_squares(P, Q)
plot_values(chi_values, label="Sum of Squared Error")
plt.show()

correspondences = corresp_values
ax = plot_data(P_values[-1], Q, "Camera data Fitted", "Gcode data Fitted")
draw_correspondeces(P_values[-1], Q, correspondences[-1], ax)
plt.show()
print('Sum of Squared Error: ', chi_values[-1])

