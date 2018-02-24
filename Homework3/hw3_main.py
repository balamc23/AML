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

# import numpy as np
# from pydatset.cifar10 import get_CIFAR10_data

# X_train, y_train, X_test, y_test = load_CIFAR10('data/cifar10/')

# print(len(X_train))

# from six.moves import cPickle as pickle
# from  PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
#
# f = open('cifar-10-batches-py/data_batch_5', 'rb')
# tupled_data= pickle.load(f, encoding='bytes')
# f.close()
# img = tupled_data[b'data']
# single_img = np.array(img[5])
# single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
# plt.imshow(single_img_reshaped)
# plt.show()

# Below is Pranav's code
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import cPickle
import pandas as pd
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

# pca = PCA(20)
# print(pca)

# print first_set['data']
# print first_set['labels']
# X = first_set['data']
# # X = StandardScaler().fit_transform(X) # This line currently gives a warning
# X = scale(X)    # should be equivalent to above line
# pca = PCA(n_components=20)
# pca.fit(X)
# var = pca.explained_variance_ratio_
# var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
# print var1
# plt.plot(var1)
# plt.show()


pca = PCA(n_components=20)
vars_arr = []
vars1_arr = []

for i in range(num_labels):
    X = mean_img_dict[i]
    X = scale(X)
    pca.fit(X)
    var = pca.explained_variance_ratio_
    var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    vars1_arr.append(var1)

for var1 in vars1_arr:
    plt.plot(var1)

plt.show()

# X_proj = pca.fit_transform(X)
# print X_proj.shape
# print "=================="
# print X_proj
# # plt.scatter(X_proj[:,0])
#
# pca = PCA().fit(X)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()

# # Features
# x = first_set['data']
# features_df = pd.DataFrame(data=x, columns=['pixel_vals' + str(i) for i in range(3072)])
#
# # Labels
# y = first_set['labels']
# labels_df = pd.DataFrame(data=y, columns=['target'])
#
# labels_targets_df = pd.concat([features_df, labels_df], axis = 1)
# print(labels_targets_df)
#
# # Sorting by label (category) value
# labels_targets_df = labels_targets_df.sort('target')
# print(labels_targets_df)
#
#
# # PCA stuff down below
# # # Standardizing/Scaling the features
# x = StandardScaler().fit_transform(x)
#
# pca = PCA(n_components=20)
# principalComponents = pca.fit_transform(x)
# print(principalComponents)
# print(len(principalComponents))
#
# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component ' + str(i) for i in range(1,21)])
#
# finalDf = pd.concat([principalDf, labels_df], axis = 1)
#
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('20 component PCA', fontsize = 20)
# targets = [i for i in range(9)]
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'C0', 'C1']
#
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['target'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()
# plt.show()

