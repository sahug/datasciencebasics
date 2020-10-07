#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:45:41 2020

@author: gajendrasahu
"""

# Let’s import the libraries first
from __future__ import absolute_import, division, print_function
# TensorFlow and tf.keras

import tensorflow as tf
from tensorflow import keras

# Helper libraries

import numpy as np
import matplotlib.pyplot as plt

#Let’s load the data
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Next, we are going to map the images into classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Exploring the data
train_images.shape

# Each Label is between 0-9
train_labels
test_images.shape

# Now, it’s time to pre-process the data.
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255.


# We have to scale the images from 0-1 to feed it into the Neural Network
train_images = train_images / 255.0

test_images = test_images / 255.0


# Let’s display some images.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# Setup the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Model Training
model.fit(train_images, train_labels, epochs=10)





