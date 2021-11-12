import tensorflow as tf
import numpy as np

def fps(xyz, npoint):
    '''
    Input:
        npoint: int
        xyz: (batch_size, ndataset, 3) TF tensor
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
    '''
    batch_size = xyz.shape[0]
    ndataset = xyz.shape[1]

    new_indices = tf.Variable(tf.zeros([batch_size, npoint,2], dtype=tf.int32))
    for ex in range(batch_size):
        new_indices[ex, : , 0].assign(tf.fill([npoint], ex))
    dist = tf.zeros([batch_size,ndataset])
    for i in range(1,npoint):
        diff = xyz-tf.reshape(tf.gather_nd(xyz, new_indices[:,i-1]),[batch_size,1,3])
        dist_to_last = tf.norm(diff, axis = -1) # shape = (batch_size, ndataset)
        dist = tf.maximum(dist, dist_to_last)
        new_indices[:,i,1].assign(tf.argmax(dist, axis = -1, output_type=tf.int32)) # shape = (batch_size)
    return tf.gather_nd(xyz, new_indices)

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    xyz = [
        [[1,0,0],[0,0,-10],[10,0,0]],
        [[-1,0,0],[0,100,0],[10,0,0]]
    ]
    xyz = tf.convert_to_tensor(xyz, dtype=tf.float32)
    new_xyz = fps(xyz, 2)
    tf.print(new_xyz)

