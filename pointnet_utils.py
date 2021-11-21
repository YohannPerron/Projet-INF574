import tensorflow as tf
from tensorflow.keras.layers import MaxPool1D, Layer, BatchNormalization

from pnet2_layers.cpp_modules import (
	farthest_point_sample,
	gather_point,
	query_ball_point,
	group_point,
	knn_point,
	three_nn,
	three_interpolate
)

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
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
    _,idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz

def pointnet_downsample(xyz, points, npoint, radius, nsample, NN, NN2 = None):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int -- #points sampled in farthest point sampling
            radius: float -- search radius in local region
            nsample: int -- how many points in each local region
            NN: list of int -- output size for MLP on each point
            NN2: list of int -- output size for MLP on each region
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    # new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points)
    new_points = xyz
    #First Dense NN on all points
    for num_out_channel in NN:
        new_points = tf.keras.layers.Conv2D(num_out_channel, (1,1), strides=(1, 1), padding='valid')(new_points)
                                                        #shape: batch_size, npoint, nsample, num_out_channel
    #max pooling
    new_points = tf.reduce_max(new_points, axis=2)      #shape: batch_size, npoint, mlp[-1]

    if NN2 !=None :
        for num_out_channel in NN2:
            new_points = tf.keras.layers.Conv2D(num_out_channel, (1,1), strides=(1, 1), padding='valid')(new_points)
                                                            #shape: batch_size, npoint, num_out_channel
    
    return new_xyz, new_points, idx



if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    xyz = [
        [[1,0,0],[0,0,-10],[10,0,0],[10,5,0]],
        [[-1,0,0],[0,100,0],[10,0,0],[10,0,7]]
    ]
    xyz = tf.convert_to_tensor(xyz, dtype=tf.float32)
    new_xyz, new_points, idx, grouped_xyz = sample_and_group(2,40,3,xyz, xyz)
    tf.print(new_points)
    

