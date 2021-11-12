import tensorflow as tf

def grouping(radius, nsample,new_xyz, xyz, points):
    '''
    Input:
        radius: float
        nsample: int
        new_xyz: (batch_size, npoint, 3)
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
    Output:
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    batch_size,ndataset,__ = xyz.shape
    npoint = new_xyz.shape[1]

    idx = tf.Variable(tf.zeros([batch_size, npoint, nsample], dtype=tf.int32))

    for centre in range(npoint):
        delta = xyz-tf.reshape(new_xyz[:,centre,:],[batch_size, 1, 3])  #shape: batch_size, ndataset, 3
        dist = tf.norm(delta, axis = -1)                                #shape: batch_size, ndataset
        sorted_indices = tf.argsort(dist, axis=-1)
        sorted = tf.gather(dist, sorted_indices, batch_dims=-1)
        centroid_indices = sorted_indices[:,0]
        y = tf.broadcast_to(centroid_indices[:, tf.newaxis],sorted_indices.shape)   #shape: batch_size, ndataset
        radius_indices = tf.where(sorted<radius, sorted_indices, y)     #shape: batch_size, ndataset
        if ndataset<nsample:
            z = tf.broadcast_to(centroid_indices[:, tf.newaxis],[batch_size, nsample-ndataset])
            radius_indices = tf.concat([radius_indices,z], axis=-1)
        idx[:,centre,:].assign(radius_indices[:,:nsample])
    
    grouped_xyz = tf.gather(xyz, idx, batch_dims=1)
    if points == None:
         return grouped_xyz, idx, grouped_xyz
    else:
        grouped_points = tf.gather(points, idx, batch_dims=1)
        new_points = tf.concat([grouped_xyz,grouped_points], axis=-1)
        return new_points, idx, grouped_xyz
