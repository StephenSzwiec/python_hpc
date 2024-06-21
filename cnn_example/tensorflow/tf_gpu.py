"""
CCAST Machine Learning Tutorial
Part 1: TensorFlow (GPU)
Author: Stephen Szwiec
Date: 2023-12-29

This file contains the code for the GPU implementation of
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

"""
Region: function definitions
"""
# function to load CIFAR-10 dataset
def load_cifar10():
    val_size = 10000
    # load CIFAR-10 dataset 
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    # convert X values to float32
    x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
    # normalize values: x = [0,255], y = [0,1]
    x_train, x_test = x_train / 255., x_test / 255.
    y_train, y_test = np.reshape(y_train, (-1)), np.reshape(y_test, (-1))
    # split training data into training and validation set 
    x_train, x_val = x_train[:-val_size], x_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]
    # convert to tensors and send to GPU
    return ( 
            tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
            tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
            tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )

# define the convolution
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
    with tf.device('/gpu:0'):
        network = cnn()
        network.compile(
            optimizer='adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        network.build(input_shape=(batch_size, 32, 32, 3))
    # print network summary
    network.summary()
    # load CIFAR-10 dataset 
    train_dataset, val_dataset, test_dataset = load_cifar10()
    # train network
    start = time.time()
    try: 
        with tf.device('/gpu:0'):
            print("Training network...")
            history = network.fit(train_dataset, epochs=10, validation_data=val_dataset)
    except RuntimeError as e:
        print(e)
    end = time.time() 
    print("Training complete!")
    print("Training time: {} seconds".format(end - start))
    print("------------------")
    # evaluate network
    print("Evaluating network on test dataset...")
    network.evaluate(test_dataset)

if __name__ == '__main__':
    main()
