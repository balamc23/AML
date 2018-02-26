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

# Calculating the mean image for each category (label)
num_labels = 10
num_pixels = 3072
labels = [i for i in range(num_labels)]
rbgs = np.zeros((num_labels, num_pixels))
labels_rbgs = zip(labels, rbgs)
mean_img_dict = dict()

for label, rbg in labels_rbgs:
    mean_img_dict[label] = rbg

# print(len(mean_img_dict[0]))
# print(len(first_set['data'][0]))
# print(mean_img_dict)
print("=========")
print(mean_img_dict[0])
print("=========")

for i in range(num_images):
    label = first_set['labels'][i]
    for j in range(num_pixels):
        mean_img_dict[label][j] += first_set['data'][i][j]

print(mean_img_dict)

for i in range(num_labels):
    for j in range(num_pixels):
        mean_img_dict[i][j] = mean_img_dict[i][j]/num_images

print(mean_img_dict)

# print('image means', mean_img_dict)

# Sorting the images by category (label)
sorted_imgs = [[] for i in range(num_labels)]
for i in range(num_images):
    label = first_set['labels'][i]
    sorted_imgs[label].append(first_set['data'][i])


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

for var1 in vars1_arr:
    plt.plot(var1)

plt.legend([str(i) for i in range(10)], loc='best')
# plt.show()


# Task 2 below
# dist_matrix = np.zeros((10, 10))
# # print(dist_matrix)
#
# for i in range(num_labels):
#   for j in range(num_labels):
#     if (j > i):
#       first_img, second_img = mean_img_dict[i], mean_img_dict[j]
#       # print('1', first_img, '2', second_img)
#       dist_matrix[i][j] = distance.euclidean(first_img, second_img)
# print(dist_matrix)
#
# dist_mat_df = pd.DataFrame(dist_matrix)
# Ar_dist = distance.squareform(distance.pdist(dist_mat_df.T))
# DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist)
# PCoA = skbio.stats.ordination.pcoa(DM_dist)
# PCoA.plot(df=dist_mat_df, column='distances')




