import tensorflow as tf
import numpy as np
import math


def get_model(cloud, is_training, bn_decay=None):
    #our output will be of the shape Bx40

    
    #our input is of is of shape BxNx3
    #B: Batch size
    #N: Number of points
    #3: number of coordinates
    Batch_Sz = cloud.get_shape()[0].value
    N_point = cloud.get_shape()[1].value


    #we instantiate out endpoints
    end_points = {}
    #we add axis -1 to the point cloud
    Input_Img = tf.expand_dims(cloud, axis=-1)
    
    #we now implement multi layer perceptrons as conv2d
    #we should have done batch normalization

    net = tf.keras.layers.Conv2D(64, (1,3), strides=(1, 1), padding='valid')(Input_Img)

    net = tf.keras.layers.Conv2D(64, (1,1), strides=(1, 1), padding='valid')(net)
   
    net = tf.keras.layers.Conv2D(64, (1,1), strides=(1, 1), padding='valid')(net)

    net = tf.keras.layers.Conv2D(128, (1,1), strides=(1, 1), padding='valid')(net)

    net = tf.keras.layers.Conv2D(1, (1,1), strides=(1, 1), padding='valid')(net)

    net = tf.keras.layers.MaxPooling2D(pool_size=(N_point, 1), strides=None, padding="valid", data_format=None)(net)

    # MLP on global point cloud vector
    net = tf.reshape(net, [Batch_Sz, -1])

    net= tf.keras.layers.Dense(512)(net)
    #add relu
    net = tf.keras.layers.ReLU()(net)

    net= tf.keras.layers.Dense(512)(net)
    net = tf.keras.layers.ReLU()(net)
    net= tf.keras.layers.Dense(256)(net)
    net = tf.keras.layers.ReLU()(net)

    net = tf.keras.layers.Dropout(0.3)(net)

    net= tf.keras.layers.Dense(40)(net)

    return net, end_points


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
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)