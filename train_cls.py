import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from pointnet2_class import get_model

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import sys
sys.setrecursionlimit(4000)

tf.random.set_seed(1234)

"""
## Load dataset
First download the file
P.S: change ModelNet10 to ModelNet40 to get same results as paper
"""

DATA_DIR = tf.keras.utils.get_file(
    "modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True,
)
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")


"""
To generate a `tf.data.Dataset()` we need to first parse through the ModelNet data
folders. 
Each folder has a category divided into train and test.

Each mesh is loaded and sampled into a point cloud.
we add the pointcloud to a list and convert it to an array (np)
We also store the current
The folder name is the label; we store it and add it the dict
"""


def parse_dataset(num_points=2048):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]

        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )


"""
Set the number of points to sample and batch size and parse the dataset.
P.S: change the number of classes to 40 for modelNet40
"""

NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

model = get_model(BATCH_SIZE,NUM_POINTS)
model.summary()

"""
Parse data in tf.data.Dataset()
"""
print("parsing data ...")
train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
    NUM_POINTS
)


"""
Augmentation includes shuffling and jittering 
"""
def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label

"""
We shuffle the dataset as it was previously ordered by class
And we apply data augmentation
"""
print("setting up training and testing datasets...")
train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

# """
# ### Build a model
# Each convolution and fully-connected layer (with exception for end layers) consits of
# Convolution / Dense -> Batch Normalization -> ReLU Activation.
# """


# def conv_bn(x, filters):
#     x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
#     x = layers.BatchNormalization(momentum=0.0)(x)
#     return layers.Activation("relu")(x)


# def dense_bn(x, filters):
#     x = layers.Dense(filters)(x)
#     x = layers.BatchNormalization(momentum=0.0)(x)
#     return layers.Activation("relu")(x)


# """
# PointNet consists of two core components. The primary MLP network, and the transformer
# net (T-net). The T-net aims to learn an affine transformation matrix by its own mini
# network. The T-net is used twice. The first time to transform the input features (n, 3)
# into a canonical representation. The second is an affine transformation for alignment in
# feature space (n, 3). As per the original paper we constrain the transformation to be
# close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).
# """


# class OrthogonalRegularizer(keras.regularizers.Regularizer):
#     def __init__(self, num_features, l2reg=0.001):
#         self.num_features = num_features
#         self.l2reg = l2reg
#         self.eye = tf.eye(num_features)

#     def __call__(self, x):
#         x = tf.reshape(x, (-1, self.num_features, self.num_features))
#         xxt = tf.tensordot(x, x, axes=(2, 2))
#         xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
#         return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


# """
#  We can then define a general function to build T-net layers.
# """


# def tnet(inputs, num_features):

#     # Initalise bias as the indentity matrix
#     bias = keras.initializers.Constant(np.eye(num_features).flatten())
#     reg = OrthogonalRegularizer(num_features)

#     x = conv_bn(inputs, 32)
#     x = conv_bn(x, 64)
#     x = conv_bn(x, 512)
#     x = layers.GlobalMaxPooling1D()(x)
#     x = dense_bn(x, 256)
#     x = dense_bn(x, 128)
#     x = layers.Dense(
#         num_features * num_features,
#         kernel_initializer="zeros",
#         bias_initializer=bias,
#         activity_regularizer=reg,
#     )(x)
#     feat_T = layers.Reshape((num_features, num_features))(x)
#     # Apply affine transformation to input features
#     return layers.Dot(axes=(2, 1))([inputs, feat_T])


# """
# The main network can be then implemented in the same manner where the t-net mini models
# can be dropped in a layers in the graph. Here we replicate the network architecture
# published in the original paper but with half the number of weights at each layer as we
# are using the smaller 10 class ModelNet dataset.
# """
# print("creating network...")
# inputs = keras.Input(shape=(NUM_POINTS, 3))
# print("adding tnet")
# x = tnet(inputs, 3)
# print("adding conv layers")
# x = conv_bn(x, 32)
# x = conv_bn(x, 32)
# print("adding tnet")
# x = tnet(x, 32)
# print("adding conv layers")
# x = conv_bn(x, 32)
# x = conv_bn(x, 64)
# x = conv_bn(x, 512)
# print("adding max pooling layer")
# x = layers.GlobalMaxPooling1D()(x)
# print("adding full connected and dropout layer")
# x = dense_bn(x, 256)
# x = layers.Dropout(0.3)(x)
# x = dense_bn(x, 128)
# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

#model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

"""
### Train model
Once the model is defined it can be trained like any other standard classification model
using `.compile()` and `.fit()`.
"""
print("compiling model")
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)
print("fitting model")
model.fit(train_dataset, epochs=20, validation_data=test_dataset)

"""
## Visualize predictions
We can use matplotlib to visualize our trained model performance.
"""

data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

# run test data through model
preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()

# score the model
score = model.evaluate(labels, preds)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])


# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(
            CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
plt.show()