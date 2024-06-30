"""
version:1

Data-Source: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Model(s)-Information:

Observations:


"""
# --------------------------------------------------------------------------------------------------

import os
import gzip
import idx2numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------

DATA_DIR = r'C:\WindowsData\ShirishPC\MachineLearning\MLProjects\MNIST\Data'

# Loading MNIST Dataset

mnist_train_images = idx2numpy.convert_from_file(os.path.join(DATA_DIR, 'train-images.idx3-ubyte'))
mnist_train_labels = idx2numpy.convert_from_file(os.path.join(DATA_DIR, 'train-labels.idx1-ubyte'))

np.shape(mnist_train_images)
# (60000, 28, 28)

np.set_printoptions(suppress=True, linewidth=100000)
mnist_train_images[0]

plt.imshow(mnist_train_images[0], cmap=plt.cm.binary)
plt.show()


# Test-Data
mnist_test_images = idx2numpy.convert_from_file(os.path.join(DATA_DIR, 't10k-images.idx3-ubyte'))
mnist_test_labels = idx2numpy.convert_from_file(os.path.join(DATA_DIR, 't10k-labels.idx1-ubyte'))

np.shape(mnist_test_images)
np.shape(mnist_test_labels)

plt.imshow(mnist_test_images[0], cmap=plt.cm.binary)
plt.show()

# --------------------------------------------------------------------------------------------------



