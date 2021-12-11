import os
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import trimesh
from matplotlib import pyplot as plt
from pointnet2_class import get_model
tf.random.set_seed(1234)


# Loading dataset
#P.S: change ModelNet10 to ModelNet40 to get same results as paper


#Downlaod file and extract zip
DATA_DIR = tf.keras.utils.get_file("modelnet.zip","http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",extract=True,)
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")


#To generate a `tf.data.Dataset()` we need to first parse through the ModelNet data folders. 
#Each folder has a category divided into train and test.
#Each mesh is loaded and sampled into a point cloud.
#we add the pointcloud to a list and convert it to an array (np)
#We also store the current
#The folder name is the label; we store it and add it the dict


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


print("parsing data ...")
TRAINPTS, TESTPTS, TRAINLABEL, TESTLABEL, CLASS_MAP = parse_dataset()

#shuffling and adding noise 
def alter(points, label):
    # adding noise
    points = points + tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffling
    points = tf.random.shuffle(points)
    return points, label


#We shuffle the dataset as it was previously ordered by class And we apply noise and shuffling
print("setting up training and testing datasets...")
TRAINSET = tf.data.Dataset.from_tensor_slices((TRAINPTS, TRAINLABEL))
TESTSET = tf.data.Dataset.from_tensor_slices((TESTPTS, TESTLABEL))

#works on 32
TRAINSET = TRAINSET.shuffle(len(TRAINPTS)).map(alter).batch(32)
TESTSET = TESTSET.shuffle(len(TESTPTS)).batch(32)



###******TNET implementation taken from Keras examples
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


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

###******


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
OUTPUTS = layers.Dense(10, activation="softmax")(x)

MODEL = keras.Model(inputs=inputs, outputs=OUTPUTS, name="pointnet")

print("compiling model")
MODEL.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)
print("fitting model")
MODEL.fit(TRAINSET, epochs=20, validation_data=TESTSET)

#Test data
TAKE = TESTSET.take(1)
PTS, LBL = list(TAKE)[0]
PTS = PTS[:8, ...]
LBL = LBL[:8, ...]

#Getting Predictions
Predictions = MODEL.predict(PTS)
Predictions = tf.math.argmax(Predictions, -1)

# Evaluate the model
Accuracy = MODEL.evaluate(LBL, Predictions)
print('Test accuracy: ', Accuracy[1])
