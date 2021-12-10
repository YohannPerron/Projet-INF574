import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from pointnet2_class import get_model

tf.random.set_seed(1234)

"""
## Load dataset
First download the file
P.S: change ModelNet10 to ModelNet40 to get same results as paper
"""

#Downlaod file and extract zip
DATA_DIR = tf.keras.utils.get_file("modelnet.zip","http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",extract=True,)
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


def parse_dataset():

    #create lsits to store train/test points and their relative labels
    trainPts = []
    trainLbl = []
    testPts = []
    testLbl = []
    CLASS_MAP = {}
  
    Directories = glob.glob(os.path.join(DATA_DIR, "[!README]*"))
    for lbl, Dir in enumerate(Directories):
        # store folder name with ID so we can retrieve later
        CLASS_MAP[lbl] = Dir.split("/")[-1]

        # gather all files inside of train and test folders
        # and classify them according to their use
        trainAll = glob.glob(os.path.join(Dir, "train/*"))
        testAll = glob.glob(os.path.join(Dir, "test/*"))

        #iterate over all files and populare lists accordingly
        for f in trainAll:
            trainLbl.append(lbl)
            trainPts.append(trimesh.load(f).sample(2048))

        for f in testAll:
            testLbl.append(lbl)
            testPts.append(trimesh.load(f).sample(2048))

    #convert list to np.array
    TRAINPTS=np.array(trainPts)
    TESTPTS=np.array(testPts)
    TRAINLABEL=np.array(trainLbl)
    TESTLABEL=np.array(testLbl)
    return (TRAINPTS,TESTPTS,TRAINLABEL,TESTLABEL,CLASS_MAP,)



"""
Parse data in tf.data.Dataset()
"""
print("parsing data ...")
TRAINPTS, TESTPTS, TRAINLABEL, TESTLABEL, CLASS_MAP = parse_dataset()


"""
shuffling and adding noise 
"""
def alter(points, label):
    # adding noise
    points = points + tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffling
    points = tf.random.shuffle(points)
    return points, label

"""
We shuffle the dataset as it was previously ordered by class
And we apply noise and shuffling
"""
print("setting up training and testing datasets...")
train_dataset = tf.data.Dataset.from_tensor_slices((TRAINPTS, TRAINLABEL))
test_dataset = tf.data.Dataset.from_tensor_slices((TESTPTS, TESTLABEL))

#use batch size of 32
train_dataset = train_dataset.shuffle(len(TRAINPTS)).map(alter).batch(32)
test_dataset = test_dataset.shuffle(len(TESTPTS)).batch(32)

"""
### Build a model
Each convolution and fully-connected layer (with exception for end layers) consits of
Convolution / Dense -> Batch Normalization -> ReLU Activation.
"""


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


"""
PointNet consists of two core components. The primary MLP network, and the transformer
net (T-net). The T-net aims to learn an affine transformation matrix by its own mini
network. The T-net is used twice. The first time to transform the input features (n, 3)
into a canonical representation. The second is an affine transformation for alignment in
feature space (n, 3). As per the original paper we constrain the transformation to be
close to an orthogonal matrix (i.e. ||X*X^T - I|| = 0).
"""


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


"""
 We can then define a general function to build T-net layers.
"""


def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


"""
The main network can be then implemented in the same manner where the t-net mini models
can be dropped in a layers in the graph. Here we replicate the network architecture
published in the original paper but with half the number of weights at each layer as we
are using the smaller 10 class ModelNet dataset.
"""
print("creating network...")
inputs = keras.Input(shape=(2048, 3))
print("adding tnet")
x = tnet(inputs, 3)
print("adding conv layers")

x = layers.Conv1D(32, kernel_size=1, padding="valid")(x)
x = layers.BatchNormalization(momentum=0.0)(x)
x = layers.Activation("relu")(x)

x = layers.Conv1D(32, kernel_size=1, padding="valid")(x)
x = layers.BatchNormalization(momentum=0.0)(x)
x = layers.Activation("relu")(x)

print("adding tnet")
x = tnet(x, 32)
print("adding conv layers")
x = layers.Conv1D(32, kernel_size=1, padding="valid")(x)
x = layers.BatchNormalization(momentum=0.0)(x)
x = layers.Activation("relu")(x)

x = layers.Conv1D(64, kernel_size=1, padding="valid")(x)
x = layers.BatchNormalization(momentum=0.0)(x)
x = layers.Activation("relu")(x)

x = layers.Conv1D(512, kernel_size=1, padding="valid")(x)
x = layers.BatchNormalization(momentum=0.0)(x)
x = layers.Activation("relu")(x)

print("adding max pooling layer")
x = layers.GlobalMaxPooling1D()(x)

print("adding full connected and dropout layer")
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

#we have 10 classes 
#P.S: change to 40 to get same results
outputs = layers.Dense(10, activation="softmax")(x)

#model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model = get_model()
model.summary()

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
