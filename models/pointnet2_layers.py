#
# Our own implementations of the layers used in pointnet++
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from pointnet2_OPS import sample_and_group, sample_and_group_all, farthest_point_sample, index_points, query_ball_point

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


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, input_size, NN_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.convolution_blocks = nn.ModuleList()
        self.batch_norm_blocks = nn.ModuleList()

        #constructs all the neural networks
        for i in range(len(NN_list)):
            convolutions = nn.ModuleList()
            batch_norms = nn.ModuleList()
            last_channel = input_size + 3 #not logical but needed for compatibility with source
            for output_size in NN_list[i]:
                convolutions.append(nn.Conv2d(last_channel, output_size, 1))
                batch_norms.append(nn.BatchNorm2d(output_size))
                last_channel = output_size
            self.convolution_blocks.append(convolutions)
            self.batch_norm_blocks.append(batch_norms)

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
        
        B, N, C = xyz.shape

        #we need to evaluate the centroid independantly from the NN we are using
        centroid_coord = index_points(xyz, farthest_point_sample(xyz, self.npoint)) #[Batch, npoint, Coord]
        new_points_list = []

        for i in range(len(self.convolution_blocks)):
            radius = self.radius_list[i]
            #only need to group the point
            grouping_index = query_ball_point(radius, K, xyz, centroid_coord)
            grouped_xyz = index_points(xyz, grouping_index)  #[Batch, npoint, nsample, Coord]
            grouped_xyz -= centroid_coord.view(B, S, 1, C) #supstract centroid coords
            #concatenat data (if exist) to coords
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            #grouped_points : [Batch, npoint, nsample, Coord+Data]

            #order grouped_point for conv2d : since kernel_size=1, only affect C ([:,C,:,:])
            grouped_points = grouped_points.permute(0, 3, 2, 1) # [Batch, Coord+Data, nsample,npoint]
            for j in range(len(self.convolution_blocks[i])):
                conv = self.convolution_blocks[i][j]
                bn = self.batch_norm_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [Batch, newData, nsample]
            new_points_list.append(new_points)

        new_points = torch.cat(new_points_list, dim=1)
        centroid_coord = centroid_coord.permute(0, 2, 1)    #[Batch, Coord, npoint]
        return centroid_coord, new_points

