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

from six.moves import cPickle as pickle
from  PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

f = open('cifar-10-batches-py/data_batch_5', 'rb')
tupled_data= pickle.load(f, encoding='bytes')
f.close()
img = tupled_data[b'data']
single_img = np.array(img[5])
single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
plt.imshow(single_img_reshaped)
plt.show()