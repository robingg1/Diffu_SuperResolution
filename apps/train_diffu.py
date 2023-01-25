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

def main(epoches, with_test=False, save_dir='checkpoints'):
    try:
        os.makedirs(os.path.join(save_dir,'body_checkpoints'))
    except:
        pass
    cuda = torch.device('cuda:%d' % 0)
    train_dataset = TrainDatasetD('/home/ai002/longbin.ji/pifuhd/recon',phase='train',device=cuda)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=8, shuffle=True)
    if with_test:
        test_dataset = TrainDatasetD('/home/ai002/longbin.ji/pifuhd/recon_tests',device=cuda)
        test_data_loader = DataLoader(test_dataset,
                                   batch_size=2, shuffle=True)

    total_loss = 0

    Diffusion = GaussianDiffusion().to(cuda)
    optimizer = torch.optim.Adam(Diffusion.parameters(), lr=0.1,  weight_decay=0.00001)

    for epoch in tqdm(range(epoches)):
       for i,(data,label) in enumerate(train_data_loader):
           if epoch%5 == 0:
            loss = Diffusion(label, data, show=True)
           else:
            loss = Diffusion(label, data)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           if (epoch*len(train_data_loader))+i%50 == 0 and epoch!=0:
            print(f'start to save checkpoint in dir {save_dir}')
            torch.save(Diffusion.state_dict(), os.path.join(save_dir,'body_checkpoints',f'model_step_{epoch*i}'))
           
           if with_test:
            if epoch%5 == 0:
               test(test_data_loader,Diffusion)

            #lr = adjust_learning_rate(optimizer, epoch, lr, opt.schedule, opt.gamma)
           print(f'loss is {loss}')

    total_loss += loss

def test(test_dataloader,model):
    total_IOU = 0
    for i, (data,label) in enumerate(test_dataloader):
        sdf = model.sample(data)
        IOU, prec, recall = compute_acc(sdf, label)
        total_IOU += IOU
        print(f'IOU: {IOU}, prec: {prec}, recall: {recall}')
    print(f'average IOU is {total_IOU/len(test_dataloader)}')


main(300)





