# Below is Bala's code
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
# with 6000 images per class. There are 50000 training images and 10000 test images. 
# The dataset is divided into five training batches and one test batch, 
# each with 10000 images. The test batch contains exactly 1000 
# randomly-selected images from each class. The training batches 
# contain the remaining images in random order, but some training batches 
# may contain more images from one class than another. Between them, 
# the training batches contain exactly 5000 images from each class.

# airplane
# automobile
# bird
# cat
# deer
# dog
# frog
# horse
# ship
# truck

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import pickle as cPickle
import numpy as np
from scipy.spatial import distance
import skbio
import pandas as pd
import math
from skbio import DistanceMatrix
from sklearn import manifold
from sklearn.metrics import euclidean_distances

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding = 'Latin1')
    return dict

with open('cifar-10-batches-py/batches.meta', 'rb') as fo:
    label_names = cPickle.load(fo)
label_names = label_names['label_names']

first_set = unpickle('cifar-10-batches-py/data_batch_1')
print(len(first_set['data'][9999]))

num_images = len(first_set['data'])
num_labels = 10
num_pixels = 3072

# Sorting the images by category (label)
sorted_imgs = [[] for i in range(num_labels)]
for i in range(num_images):
    label = first_set['labels'][i]
    sorted_imgs[label].append(first_set['data'][i])

# print(sorted_imgs)

# Calculating the mean image for each category (label)
labels = [i for i in range(num_labels)]
rbgs = np.zeros((num_labels, num_pixels))
labels_rbgs = zip(labels, rbgs)
mean_img_dict = dict()

for label, rbg in labels_rbgs:
    mean_img_dict[label] = rbg

for i in range(num_labels):
    mean_img_dict[i] = np.mean(sorted_imgs[i], axis=0)

print('image means', mean_img_dict)


# PCA stuff below
pca = PCA(n_components=20)
vars_arr = []
vars1_arr = []

for i in range(num_labels):
    X = sorted_imgs[i]
    X = scale(X)
    pca.fit(X)
    var = pca.explained_variance_ratio_
    var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    vars1_arr.append(var1)

# Uncomment bottom 4 lines to show Task 1 plot
# for var1 in vars1_arr:
#     plt.plot(var1)
#
# plt.legend([str(i) for i in range(10)], loc='best')
# plt.show()


# # Task 2 below
dist_matrix = np.zeros((10, 10))
for i in range(num_labels):
  for j in range(num_labels):
      '''
       Only computing values that haven't been already computed (avoiding redundancies)
       Might have to change this because they want us to include our 10x10 distance
       matrix in our report
      '''
      if (j > i):
        dist_matrix[i][j] = math.sqrt(np.sum((mean_img_dict[i] - mean_img_dict[j])**2))
print("newest implementation")
print(dist_matrix)


def reshape_2D(mean_image_dist_arr):
    mds = manifold.MDS(n_components = 2)
    scaled_down = mds.fit_transform(mean_image_dist_arr)
    return scaled_down

should_plot_this = reshape_2D(dist_matrix)
print(should_plot_this)

plt.scatter(should_plot_this[:, 0], should_plot_this[:, 1],
            color='darkorange', s=100, lw=0, label='NMDS')
plt.show()
