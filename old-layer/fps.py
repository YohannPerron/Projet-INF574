import tensorflow as tf

def fps(xyz, npoint):
    '''
    Input:
        npoint: int
        xyz: (batch_size, ndataset, 3) TF tensor
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
    '''
    batch_size,ndataset,__ = xyz.shape

    new_indices = tf.zeros([batch_size, 1], dtype=tf.int32)

    dist = tf.zeros([batch_size,ndataset])
    for i in range(1,npoint):
        delta = xyz-tf.reshape(tf.gather(xyz, new_indices[:,i-1], batch_dims=1),[batch_size,1,3])
        dist_to_last = tf.norm(delta, axis = -1) # shape = (batch_size, ndataset)
        dist = tf.maximum(dist, dist_to_last)
        new_indices = tf.concat([new_indices,tf.argmax(dist, axis = -1, output_type=tf.int32)[:, tf.newaxis]], axis = -1) # shape = (batch_size)
    return tf.gather(xyz, new_indices, batch_dims=1)

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    xyz = [
        [[1,0,0],[0,0,-10],[10,0,0]],
        [[-1,0,0],[0,100,0],[10,0,0]]
    ]
    xyz = tf.convert_to_tensor(xyz, dtype=tf.float32)
    new_xyz = fps(xyz, 2)
    tf.print(new_xyz)

class fpsLayer(tf.keras.layers.Layer):
    def __init__(self, batch_size, ndataset, npoint):
        super(fpsLayer, self).__init__()
        self.batch_size = batch_size
        self.ndataset = ndataset
        self.npoint = npoint
        self.new_indices = tf.Variable(
            initial_value=tf.zeros([batch_size, npoint], dtype=tf.int32), trainable=False
        )

    def call(self, xyz):
        dist = tf.zeros([self.batch_size,self.ndataset])
        for i in range(1,self.npoint):
            delta = xyz-tf.reshape(tf.gather(xyz, self.new_indices[:,i-1], batch_dims=1),[self.batch_size,1,3])
            dist_to_last = tf.norm(delta, axis = -1) # shape = (batch_size, ndataset)
            dist = tf.maximum(dist, dist_to_last)
            self.new_indices[:,i].assign(tf.argmax(dist, axis = -1, output_type=tf.int32)) # shape = (batch_size)
        return tf.gather(xyz, self.new_indices, batch_dims=1)