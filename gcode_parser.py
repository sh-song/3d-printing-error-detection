from parse import *
import os
import numpy as np
import matplotlib.pyplot as plt

class Parser:
    def __init__(self, \
                data_path, \
                file_name, \
                edited_file_path_n_name, \
                slicing_times, \
                layer_image_path):


        self.data_path = data_path
        self.file_name = file_name
        self.edited_file_path_n_name = edited_file_path_n_name
        self.slicing_times = slicing_times
        self.targeted_layer_num = -1
        self.layer_image_path = layer_image_path

        self.points_x = np.array([])
        self.points_y = np.array([])
        self.points_z = np.array([])

        self.sliced_xys = {}
        self.x_len = -1

        self.parsing_end_check = False
    def input_file(self):

        self.path = os.path.join(self.data_path,self.file_name)

        self.file_for_count = open(self.path)
        self.file_main = open(self.path) 

    def count_points(self):
        print('counting command lines...')
        controlled = 0
        for line in self.file_for_count.readlines():
            if 'G1' in line:
                controlled += 1
        #counting number of points        

        print('total {} points'.format(controlled))

#---parsing-------------------------------------
    def parse(self):
        x = 0
        y = 0
        z = 0
        is_printing_wall_outer = 0
        self.layer_number = 0
        self.layer_number_bank = np.array([])
        for line in self.file_main.readlines():
            if 'LAYER:' in line:
                self.layer_number += 1
            if 'TYPE:' in line:
                if 'WALL-OUTER' or 'WALL-INNER' in line:
                    is_printing_wall_outer = 1
                else:
                    is_printing_wall_outer = 0
            if 'Z' in line:
                z_num = search('Z{:g}',line) #{:g} : real number
                if z_num == None:
                    continue
                z = float(z_num[0]) #extracting Z cordinate

            if is_printing_wall_outer == 1:    
                if 'G1' in line:
                    result = search('X{:g} Y{:g}',line)
                    if result == None:
                        continue
                    if result[0] == 0.0 or result[1] == 5.0:
                        continue

                    x = float(result[0]) #extracting X cordinate
                    y = float(result[1]) #extracting Y cordiante

                    self.points_x = np.append(self.points_x,[x]) 
                    self.points_y = np.append(self.points_y,[y])
                    self.points_z = np.append(self.points_z,[z])
                    self.layer_number_bank = np.append(self.layer_number_bank, self.layer_number)
        print(self.layer_number_bank)
        print(self.layer_number)

    def slicing(self, targeted_layer_num, vis=False):
        self.targeted_layer_num = targeted_layer_num
        cnt_forward = 0
        cnt_backward = 0
        sliced_x = np.array([])
        sliced_y = np.array([])
        sliced_z = np.array([])
        i = 0
        if self.targeted_layer_num == self.layer_number:
            while self.layer_number_bank[i] < self.targeted_layer_num:
                cnt_forward += 1
                i += 1

            cnt_backward = len(self.layer_number_bank)
        else:
            while self.layer_number_bank[i] < self.targeted_layer_num+1:
                cnt_backward += 1
                if self.layer_number_bank[i] < self.targeted_layer_num:
                    cnt_forward = cnt_backward
                i += 1

        sliced_x = self.points_x[cnt_forward:cnt_backward]
        sliced_y = self.points_y[cnt_forward:cnt_backward]
        sliced_z = self.points_z[cnt_forward:cnt_backward]

        edges={}

        try:
            x_len = len(sliced_x)

            edges['right'] = np.max(self.points_x[3:])
            edges['left'] = np.min(self.points_x[3:])
            edges['top'] = np.max(self.points_y[3:])
            edges['bottom'] = np.min(self.points_y[3:])

            layer_sliced_xys = np.zeros((x_len, 2))

            for i in range(x_len):
                temp = [sliced_x[i], sliced_y[i]]
                layer_sliced_xys[i] = temp

            return layer_sliced_xys, edges

        except:
            ValueError
        
        return None, None
    def start(self):
        self.parse()
        self.edit_gcode()
        max_x = 0
        min_x = 99999
        max_y = 0
        min_y = 99999
        plt.figure(figsize=(15,15))
        for i in range(self.slicing_times):
            layer_name = self.exe_times * i
            result, edges= self.slicing(layer_name)
             
            if result is not None:
                self.save(layer_name, result, edges)
                self.sliced_xys[str(layer_name)] = result
            
        print('============max_x', max_x, min_x, max_y, min_y) 

        
    
        print(self.sliced_xys.keys())
        self.parsing_end_check = True

        print('------------------end')



    def edit_gcode(self):

        f = open(self.path,'r+')
        f_write = open(self.edited_file_path_n_name,'w+')


        layer_count = self.layer_number
        self.exe_times = layer_count // self.slicing_times
        lines = f.readlines()

        for j,line in enumerate(lines):
            f_write.write(line)
            if 'LAYER:' in line:
                result = search('LAYER:{:g}',line)
                for i in range(1,self.exe_times):
                    if result[0] == i * self.exe_times:
                        f_write.write('G0 F6000 X0.0 Y200.0\n')
                        f_write.write(lines[j-2])

        f_write.close()

    def crop(self):
        self.top = np.max(self.sliced_y)
        self.bottom = np.min(self.sliced_y)
        self.left = np.min(self.sliced_x)
        self.right = np.max(self.sliced_x)
      
        print(self.top,self.bottom,self.left,self.right)


    def save(self, layer_name, layer_array, edges):

        sliced_x = layer_array.T[0]
        sliced_y = layer_array.T[1]
        
        # plt.figure(figsize = (10,10))
        # ax = plt.subplot(1,1,1,projection = '3d')
        # ax.plot(self.sliced_x,self.sliced_y,self.sliced_z,'k-')
        # plt.show()


        
        

        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.plot(sliced_x,sliced_y,'k-',linewidth=1)
        plt.axis((edges['left']-1,edges['right']+1,edges['bottom']-1,edges['top']+1))
        plt.axes().set_aspect('equal')
        name = 'No_%i_Layer.png'%self.targeted_layer_num
        path = os.path.join(self.layer_image_path, name)
        plt.savefig(path, bbox_inches='tight', pad_inches = 0)
        print(edges)
        print('save success')
