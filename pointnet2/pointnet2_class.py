import tensorflow as tf
import numpy as np
from pointnet_utils import pointnet_downsample
import math
import sys
import os

def get_model(point_cloud):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size, num_point ,_ = point_cloud.shape

    xyz_0 = point_cloud
    points_0 = None

    # downsampling
    xyz_1, points_1, _ = pointnet_downsample(xyz_0, points_0, 512, 0.2, 32, [64,64,128])
    xyz_2, points_2, _ = pointnet_downsample(xyz_1, points_1, 128, 0.4, 64, [128,128,256])
    xyz_3, points_3, _ = pointnet_downsample(xyz_2, points_2, 1, np.inf, 128, [256,512,1024])

    # Fully connected layers
    net = tf.reshape(points_3, [batch_size, -1])
    net= tf.keras.layers.Dense(512)(net)
    net = tf.keras.layers.ReLU()(net)
    net = tf.keras.layers.Dropout(0.5)(net)
    net= tf.keras.layers.Dense(256)(net)
    net = tf.keras.layers.ReLU()(net)
    net = tf.keras.layers.Dropout(0.5)(net)
    net= tf.keras.layers.Dense(40)(net)


    return net


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        net = get_model(inputs)
        print(net)