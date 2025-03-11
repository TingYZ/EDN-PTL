from __future__ import division
import torch
import torch.nn as nn
from utils import outputActivation


class tcnFusion(nn.Module):
    def __init__(self):
        super(tcnFusion, self).__init__()

    def forward(self, x):
        """
        堆叠因果卷积，不同层特征融合
        :param x: (b_s, c, len)
        :return:
        """


class highwayNet(nn.Module):

    # Initialization
    def __init__(self,args):
        super(highwayNet, self).__init__()


    # Forward Pass
    def forward(self,hist, nbrs, masks):
        """
        hist: (16, b_s, 2)
        nbrs: (16, nbr_num_all_batch, 2)
        masks: (b_s, 3, 13, encoder_size)
        """


