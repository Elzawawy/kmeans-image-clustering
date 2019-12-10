from tqdm import tqdm
import os
import pickle
import numpy as np
from utils import unpickle_to_dict

class ImageDataLoader(object):
    """"""
    def __init__(self):
          super().__init__()

    def load_cifar10(self, data_dir, num_batches = 5):
        """"""
        train_X = []      # training data to be returned.
        train_y = []      # training labels to be returned.
        expected_listdir = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'readme.html', 'test_batch']
       
        if(num_batches <= 0 or num_batches > 5):
            raise ValueError("Number of Batches must be in range of [1..5].")
        if(not os.path.isdir(data_dir)):
            raise NotADirectoryError("Data Directory must be a directory. Passed argument is not a directory or doesn't exist.")
        if(not sorted(os.listdir(data_dir)) == sorted(expected_listdir)):
            raise ValueError("Directroy passed is not the CIFAR-10 dataset directory. May be caused by wrong or missing files.")

        train_batch_list = sorted([x for x in os.listdir(data_dir) if(x.split("_")[0] == 'data')])
        for batch in tqdm(train_batch_list[:num_batches]):
            train_batch_dict = unpickle_to_dict(os.path.join(data_dir,batch))
            train_X.extend(train_batch_dict[b'data'])
            train_y.extend(train_batch_dict[b'labels'])

        train_X = np.array(train_X)
        train_X = train_X.reshape(train_X.shape[0], 3, 32, 32)
        train_y = np.array(train_y)

        test_batch_dict = unpickle_to_dict(os.path.join(data_dir,'test_batch'))
        test_X = test_batch_dict[b'data']
        test_X = test_X.reshape(test_X.shape[0], 3, 32, 32)
        test_y = np.array(test_batch_dict[b'labels'])

        return train_X, train_y, test_X, test_y