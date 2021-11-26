#
# Our own implementations of the layers used in pointnet++
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from pointnet2_OPS import sample_and_group, sample_and_group_all, farthest_point_sample

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, input_size, NNlayers, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        #construct the neural network
        self.NN_convolutions = nn.ModuleList()
        self.NN_batch_norms = nn.ModuleList()
        last_conv_size = input_size
        for output_size in NNlayers:
            self.NN_convolutions.append(nn.Conv2d(last_conv_size, output_size, 1))
            self.NN_batch_norms.append(nn.BatchNorm2d(output_size))
            last_conv_size = output_size

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [Batch, Coord, Nb_point]
            points: input points data, [Batch, Data, Nb_point]
        Return:
            new_xyz: sampled points position data, [Batch, Coord, nsample]
            new_points_concat: sample points feature data, [Batch, newData, nsample]
        """
        #reorder inputs for futur conv
        xyz = xyz.permute(0, 2, 1)  #[Batch, Nb_point, Coord]
        if points is not None:
            points = points.permute(0, 2, 1) #[Batch, Nb_point, Data]

        #sample the point cloud
        if self.group_all:
            centroid_coord, grouped_points = sample_and_group_all(xyz, points)
        else:
            centroid_coord, grouped_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # centroid_coord [Batch, npoint, Coord]
        # grouped_points [Batch, npoint, nsample, Coord+Data]
        #order grouped_point for conv2d : since kernel_size=1, only affect C ([:,C,:,:])
        grouped_points = grouped_points.permute(0, 3, 2, 1) # [Batch, Coord+Data, nsample,npoint]
        for i in range(len(self.NN_convolutions)):
            batch_norm = self.NN_batch_norms[i]
            convolution = self.NN_convolutions[i]
            grouped_points =  F.relu(batch_norm(convolution(grouped_points)))

        grouped_points = torch.max(grouped_points, 2)[0]
        centroid_coord = centroid_coord.permute(0, 2, 1)    #[Batch, Coord, npoint]
        return centroid_coord, grouped_points

