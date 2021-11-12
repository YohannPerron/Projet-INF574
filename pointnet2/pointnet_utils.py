import tensorflow as tf
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
    new_xyz = fps.fps(xyz, npoint)
    new_points, idx, grouped_xyz = grouping.grouping(radius, nsample,new_xyz, xyz, points)
    return new_xyz, new_points, idx, grouped_xyz

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    xyz = [
        [[1,0,0],[0,0,-10],[10,0,0],[10,5,0]],
        [[-1,0,0],[0,100,0],[10,0,0],[10,0,7]]
    ]
    xyz = tf.convert_to_tensor(xyz, dtype=tf.float32)
    new_xyz, new_points, idx, grouped_xyz = sample_and_group(2,40,3,xyz, xyz)
    tf.print(new_points)
    
