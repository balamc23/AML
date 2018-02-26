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

# print(len(final_data_set))

final_label_set = np.append(first_set['labels'],second_set['labels'],axis = 0)
final_label_set = np.append(final_label_set,third_set['labels'],axis = 0)
final_label_set = np.append(final_label_set,fourth_set['labels'],axis = 0)
final_label_set = np.append(final_label_set,fifth_set['labels'],axis = 0)
final_label_set = np.append(final_label_set,sixth_set['labels'],axis = 0)

# print(len(final_label_set))

first_set['data'] = final_data_set
first_set['labels'] = final_label_set

print(len(first_set['data']))
print(len(first_set['labels']))

# for i in range(len(second_set['data'])):
# 	first_set['data'].append(second_set['data'][i])


# for key, value in second_set.items():
# 	first_set.setdefault(key, []).extend(value)


# print(len(first_set['data']))
# print(len(first_set['labels']))



# num_images = len(first_set['data']) + len(second_set['data']) + len(third_set['data']) + len(fourth_set['data']) + len(fifth_set['data'])
# 				+ len(sixth_set['data'])
# num_labels = 10
# num_pixels = 3072


# sorted_imgs = [[] for i in range(num_labels)]
# for i in range(num_images):
#     label1= first_set['labels'][i]
#     sorted_imgs[label].append(first_set['data'][i])
#     label1= first_set['labels'][i]
#     sorted_imgs[label].append(first_set['data'][i])





