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
pcas_arr = []
vars_arr = []
vars1_arr = []

for i in range(num_labels):
    X = sorted_imgs[i]
    pca.fit(X)
    pcas_arr.append(pca)
    var = pca.explained_variance_ratio_
    var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    vars1_arr.append(var1)

    print('1. var', var)
    print('2. var1', var1)
    print('3. sum', np.sum(var1))
# Uncomment bottom 4 lines to show Task 1 plot
plt.figure(1)
for var1 in vars1_arr:
    plt.plot(var1)

plt.legend([label_names[i] for i in range(10)], loc='best')
plt.xlabel('Number of Principal Components used')
plt.ylabel('Accuracy')


# Task 2 below
dist_matrix = np.zeros((10, 10))
for i in range(num_labels):
    for j in range(num_labels):
        dist_matrix[i][j] = math.sqrt(np.sum((mean_img_dict[i] - mean_img_dict[j])**2))
print("newest implementation")
print(dist_matrix)


def reshape_2D(mean_image_dist_arr):
    mds = manifold.MDS(n_components = 2)
    scaled_down = mds.fit_transform(mean_image_dist_arr)
    return scaled_down

should_plot_this = reshape_2D(dist_matrix)

plt.figure(2)
x,y = zip(*should_plot_this)
plt.scatter(x,y)
i = 0
for ab in zip(x,y):
    plt.annotate(label_names[i], xy=ab,textcoords='data')
    i+=1

plt.title('PCoA 2D Map of Means of Each Category')
plt.grid()
plt.show()


# Task 3 below
# Calculating error with using mean image
err_by_categ = []

for i in range(num_labels):
    curr_categ_mean_img = mean_img_dict[i]
    curr_categ_imgs = sorted_imgs[i]
    categ_err = 0
    for j in range(len(curr_categ_imgs)):
        curr_err = abs(np.mean(curr_categ_mean_img - curr_categ_imgs[j]))
        categ_err += curr_err
        print(curr_err)
    err_by_categ.append(categ_err)

print(err_by_categ)

# Calculating
for i in range(num_labels):
    curr_pca = pcas_arr[i]
    for j in range(num_labels):
        if j > i:
            other_categ_imgs = sorted_imgs[j]
            other_categ_imgs_pca = curr_pca.transform(other_categ_imgs)
            projected = curr_pca.inverse_transform(other_categ_imgs_pca)
            print('length', len(projected))
            print('projected', projected)










