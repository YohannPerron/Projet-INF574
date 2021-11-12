import tensorflow as tf
import numpy as np
import math
import sys
import os
import fps
import grouping

def sample_and_group(npoint, radius, nsample, xyz, points):
    '''
    Input:
        npoint: int
        radius: float
        nsample: int
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    new_xyz = fps(xyz)
    
