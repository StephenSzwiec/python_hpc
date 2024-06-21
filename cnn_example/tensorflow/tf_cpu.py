"""
CCAST Machine Learning Tutorial
Part 1: TensorFlow (CPU)
Author: Stephen Szwiec
Date: 2023-12-29

This file contains the code for the CPU implementation of
classification of the CIFAR-10 dataset using TensorFlow. 
The code is based on The TensorFlow Authors' tutorial: 
https://www.tensorflow.org/tutorials/images/cnn
"""

"""
Region: import libraries
"""
import os
import time 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

"""
Region: parameters
"""
# training parameters 
learning_rate = 0.001
epochs = 5
batch_size = 500
num_classes = 10 

# network parameters
conv1_filters = 64
conv2_filters = 128
conv3_filters = 256

# session parameters
numThreads = int(os.environ['NCPUS'])
if not numThreads:
    numThreads = 1 
numInterOpThreads = 1 
numIntraOpThreads = int(numThreads)
os.environ['OMP_NUM_THREADS'] = str(numThreads)
tf.config.threading.set_inter_op_parallelism_threads(numInterOpThreads)
tf.config.threading.set_intra_op_parallelism_threads(numIntraOpThreads)
print("Number of threads: ", numThreads)

"""
Region: function definitions
"""
# function to load CIFAR-10 dataset
@tf.function
def load_cifar10():
    val_size = 10000
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    # convert X values to float32
    x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
    # normalize values: x = [0,255], y = [0,1]
    x_train, x_test = x_train / 255., x_test / 255.
    y_train, y_test = np.reshape(y_train, (-1)), np.reshape(y_test, (-1))
    x_train, x_val = x_train[:-val_size], x_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]
    # create datasets
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch_size).prefetch(1),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(len(x_val)).batch(batch_size).prefetch(1),
        tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(len(x_test)).batch(batch_size).prefetch(1)
    )

# define the convolutional neural network
def cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(conv1_filters, kernel_size=3, padding='SAME', activation='relu'))
    model.add(layers.Conv2D(conv1_filters, kernel_size=3, padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D(2, strides=2))
    model.add(layers.Conv2D(conv2_filters, kernel_size=3, padding='SAME', activation='relu'))
    model.add(layers.Conv2D(conv2_filters, kernel_size=3, padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D(2, strides=2))
    model.add(layers.Conv2D(conv3_filters, kernel_size=3, padding='SAME', activation='relu'))
    model.add(layers.Conv2D(conv3_filters, kernel_size=3, padding='SAME', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(num_classes))
    return model 

"""
Region: main
"""
def main():
    # print TF version
    print("TensorFlow version: {}".format(tf.__version__))
    # create network 
    network = cnn()
    network.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    network.build(input_shape=(batch_size, 32, 32, 3))
    network.summary()
    # load CIFAR-10 dataset 
    train_dataset, val_dataset, test_dataset = load_cifar10()
    # train network
    start = time.time()
    history = network.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
    end = time.time()
    print("Training complete!")
    print("Training time: {} seconds".format(end - start))
    print("------------------")
    # evaluate network
    print("Evaluating network on test dataset...")
    network.evaluate(test_dataset)

if __name__ == '__main__':
    main()

