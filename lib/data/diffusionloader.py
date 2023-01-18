from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch

class TrainDatasetD(Dataset):
    def __init__(self, input_path, phase='train'):
        self.input_path = input_path
        self.bodies = os.listdir(input_path)
        self.point_offset = 'point'
        self.sdf = 'sdf'
        self.points_person = 50


    def __len__(self):
        return len(self.bodies)*self.points_person

    def get_item(self,index):
        body_name = index//50
        point_num = index%50
        print(index, body_name,point_num)
        points_corrad = np.load(os.path.join(self.input_path, self.bodies[body_name] ,self.point_offset+'.npy'))
        sdfs = np.load(os.path.join(self.input_path, self.bodies[body_name], self.sdf+'.npy'))
        print(sdfs.shape)
        
        point_tensor = torch.FloatTensor(np.array(points_corrad[point_num]))
        sdf_tensor = torch.FloatTensor(np.array(sdfs[point_num]))
        print(point_tensor.shape)
        return point_tensor, sdf_tensor


    def __getitem__(self, index):
        return self.get_item(index)