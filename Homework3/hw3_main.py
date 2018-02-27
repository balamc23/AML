from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle as cPickle
import numpy as np
import math
from sklearn import manifold

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding = 'Latin1')
    return dict

with open('cifar-10-batches-py/batches.meta', 'rb') as fo:
    label_names = cPickle.load(fo)
label_names = label_names['label_names']

first_set = unpickle('cifar-10-batches-py/data_batch_1')
second_set = unpickle('cifar-10-batches-py/data_batch_2')
third_set = unpickle('cifar-10-batches-py/data_batch_3')
fourth_set = unpickle('cifar-10-batches-py/data_batch_4')
fifth_set = unpickle('cifar-10-batches-py/data_batch_5')
sixth_set = unpickle('cifar-10-batches-py/test_batch')

final_data_set = np.append(first_set['data'],second_set['data'],axis = 0)
final_data_set = np.append(final_data_set,third_set['data'],axis = 0)
final_data_set = np.append(final_data_set,fourth_set['data'],axis = 0)
final_data_set = np.append(final_data_set,fifth_set['data'],axis = 0)
final_data_set = np.append(final_data_set,sixth_set['data'],axis = 0)

final_label_set = np.append(first_set['labels'],second_set['labels'],axis = 0)
final_label_set = np.append(final_label_set,third_set['labels'],axis = 0)
final_label_set = np.append(final_label_set,fourth_set['labels'],axis = 0)
final_label_set = np.append(final_label_set,fifth_set['labels'],axis = 0)
final_label_set = np.append(final_label_set,sixth_set['labels'],axis = 0)

final_set = dict()
final_set['data'] = final_data_set
final_set['labels'] = final_label_set

num_images = len(final_set['data'])
num_labels = 10
num_pixels = 3072

# Sorting the images by category (label)
sorted_imgs = [[] for i in range(num_labels)]
for i in range(num_images):
    label = final_set['labels'][i]
    sorted_imgs[label].append(final_set['data'][i])

# Calculating the mean image for each category (label)
labels = [i for i in range(num_labels)]
rbgs = np.zeros((num_labels, num_pixels))
labels_rbgs = zip(labels, rbgs)
mean_img_dict = dict()

for label, rbg in labels_rbgs:
    mean_img_dict[label] = rbg

for i in range(num_labels):
    mean_img_dict[i] = np.mean(sorted_imgs[i], axis=0)

# PCA stuff below
pcas_arr = []
vars1_arr = []

for i in range(num_labels):
    X = sorted_imgs[i]
    pca = PCA(n_components=20)
    pca = pca.fit(X)
    pcas_arr.append(pca)
    var = pca.explained_variance_ratio_
    var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    vars1_arr.append(var1)

plt.figure(1)
for var1 in vars1_arr:
    plt.plot(var1)

plt.legend([label_names[i] for i in range(10)], loc='best')
plt.xlabel('Number of Principal Components used')
plt.ylabel('Accuracy')
# locs, labels = plt.xticks()
# plt.xlim(0, 20)
plt.xticks( [0, 5, 10, 15, 20], ('0', '5', '10', '15', '20'))
plt.xlim(0, 20)
plt.ylim(0, 100)
plt.title('Accuracy versus Number of Principal Components Used')


# Task 2 below
dist_matrix = np.zeros((10, 10))
for i in range(num_labels):
    for j in range(num_labels):
        dist_matrix[i][j] = math.sqrt(np.sum((mean_img_dict[i] - mean_img_dict[j])**2))
print('Distance matrix below')
print(dist_matrix)
print('=========================================')


def reshape_2D(dist_arr):
    mds = manifold.MDS(n_components=2)
    mds_trans = mds.fit_transform(dist_arr)
    return mds_trans

should_plot_this = reshape_2D(dist_matrix)

plt.figure(2)
x,y = zip(*should_plot_this)
plt.scatter(x,y)
i = 0
for ab in zip(x,y):
    plt.annotate(label_names[i], xy=ab,textcoords='data')
    i+=1

plt.title('PCoA 2D Map of Means of Each Category')


# Task 3 below
# Calculating error from using class B's principal components to
# represent the original images of class A
errs_matrix = np.zeros((10, 10))
for i in range(num_labels):
    curr_pca = pcas_arr[i]
    for j in range(num_labels):
        other_categ_imgs = sorted_imgs[j]
        other_categ_imgs_pca = curr_pca.transform(other_categ_imgs)
        projected = curr_pca.inverse_transform(other_categ_imgs_pca)
        other_categ_imgs = np.asarray(other_categ_imgs)
        error = np.sum((other_categ_imgs - projected)**2)
        errs_matrix[i][j] = error
        # print('error for categories', label_names[i], label_names[j], error)
print('Error matrix below')
print(errs_matrix)
print('=========================================')

# Similarity matrixL
simil_matrix = np.zeros((10, 10))
for i in range(num_labels):
    for j in range(num_labels):
        err_i_j = errs_matrix[i][j]
        err_j_i = errs_matrix[j][i]
        simil_matrix[i][j] = (err_i_j + err_j_i)/2
print('Similarity matrix below')
print(simil_matrix)
print('=========================================')

# print('Reshaped 2D Similarity matrix below')
# reshaped_simil_matrix = reshape_2D(simil_matrix)
# print(reshaped_simil_matrix)
# print('=========================================')
#
# plt.figure(3)
# x,y = zip(*reshaped_simil_matrix)
# plt.scatter(x, y)
# i = 0
# for ab in zip(x,y):
#     plt.annotate(label_names[i], xy=ab,textcoords='data')
#     i+=1
#
# plt.xlim(-10000000000, 10000000000)
# plt.ylim(-10000000000, 10000000000)
# plt.title('Similarity Measures for Each Category')
# plt.show()





