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
import cPickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

with open('cifar-10-batches-py/batches.meta', 'rb') as fo:
    label_names = cPickle.load(fo)
label_names = label_names['label_names']

first_set = unpickle('cifar-10-batches-py/data_batch_1')
print(len(first_set['data'][9999][0:1024]))

num_images = len(first_set['data'])

# Calculating the mean image for each category (label)
num_labels = 10
labels = [i for i in range(10)]
rbgs = np.zeros((10, 3))
labels_rbgs = zip(labels, rbgs)
mean_img_dict = dict()

for label, rbg in labels_rbgs:
    mean_img_dict[label] = rbg

# print(mean_img_dict)

for i in range(num_images):
    label = first_set['labels'][i]
    red_vals = np.asarray(first_set['data'][i][0:1024])
    blu_vals = np.asarray(first_set['data'][i][1025:2048])
    grn_vals = np.asarray(first_set['data'][i][2049:3072])

    mean_img_dict[label][0] += np.mean(red_vals)
    mean_img_dict[label][1] += np.mean(blu_vals)
    mean_img_dict[label][2] += np.mean(grn_vals)

for i in range(10):
    for j in range(3):
        mean_img_dict[i][j] = mean_img_dict[i][j]/num_images

print(mean_img_dict)


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
plt.show()







