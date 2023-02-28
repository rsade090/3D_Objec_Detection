import kitti_util as utils
import kitti_object
import numpy as np
import os
import argparse







#Create_Top

def create_TopView(output_dir):
    #import kitti_object
    dtype = np.float32
    n_vec = 4 
    #pc_dir = 

    #my_dataset=[2,3]
    my_dataset = kitti_object.kitti_object("/home/sadeghianr/Desktop/Datasets/Kitti/",split="training")
    print("len mydataste is ",len(my_dataset))
    
    
    for data_idx in range(len(my_dataset)):

        pc_vel = my_dataset.get_lidar(data_idx,dtype,n_vec)[:,0:n_vec]

        top_view = utils.lidar_to_top(pc_vel)

        reshaped_topview=top_view.reshape(top_view.shape[0],-1)
        np.savetxt(output_dir+str(data_idx)+'.txt',reshaped_topview)
    #np.savetxt('/home/sadeghianr/Desktop/preprocessed_Kittidata/Topview/data.txt',reshaped_topview)
    
    #return n_vec+output_dir   

def Load_saved_Top_text(input_dir):
    loaded_top_txt = np.loadtxt(input_dir)
  
# This loadedArr is a 2D array, therefore
# we need to convert it to the original
# array shape.reshaping to get original
# matrice with original shape.
    load_original_arr = loaded_top_txt.reshape(
    loaded_top_txt.shape[0], loaded_top_txt.shape[1] // 15, 15 )

    print(load_original_arr.shape)


#Create FoV
def create_Fov()