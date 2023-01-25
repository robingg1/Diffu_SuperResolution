from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch

class TrainDatasetD(Dataset):
    def __init__(self, input_path, phase='train',device = 'cpu'):
        self.input_path = input_path
        self.bodies = os.listdir(input_path)
        self.point_offset = 'feat'
        self.sdf = 'sdf'
        self.points_person = 18
        self.device = device


    def __len__(self):
        return len(self.bodies)*self.points_person

    def get_item(self,index):
        body_name = index//self.points_person
        point_num = index%self.points_person
        points_corrad = np.load(os.path.join(self.input_path, self.bodies[body_name] ,self.point_offset+'.npy'))
        sdfs = np.load(os.path.join(self.input_path, self.bodies[body_name], self.sdf+'.npy'))
        
        point_tensor = torch.FloatTensor(np.array(points_corrad[point_num])).to(self.device)
        sdf_tensor = torch.FloatTensor(np.array(sdfs[point_num])).to(self.device)
        return point_tensor, sdf_tensor


    def __getitem__(self, index):
        return self.get_item(index)