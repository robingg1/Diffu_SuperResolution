import sys
import os
import time
import json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import cv2
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.geometry import index

def main(epoches):
    cuda = torch.device('cuda:%d' % 0)
    train_dataset = TrainDatasetD('/home/surf/data/longbin.ji/cmu/pifuhd/recon',phase='train')
    #test_dataset = TrainDatasetD('test')

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=2, shuffle=True,
                                   num_workers=8, pin_memory=True)

    total_loss = 0

    Diffusion = GaussianDiffusion()
    optimizer = torch.optim.RMSprop(Diffusion.parameters(), lr=0.05, momentum=0, weight_decay=0.001)
    for epoch in tqdm(range(epoches)):
       for data,label in enumerate(train_data_loader):
           print(label.shape)
           loss = Diffusion(label, data)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)
           print(f'loss is {loss}')

    total_loss += loss

main(300)





