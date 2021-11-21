import os
import glob
import trimesh
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from pointnet2_class import get_model
import models.cls_msg_model as model

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()


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
BATCH_SIZE = 1

# model = get_model(NUM_POINTS)
model = model.CLS_MSG_Model(BATCH_SIZE,NUM_CLASSES)
# model.summary()

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