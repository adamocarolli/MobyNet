from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import glob

from skimage import transform, io
import numpy as np

from tqdm import tqdm


def read_whale_flukes():
    """Read whale flukes dataset, save to a single npy file"""
    train = './data/train'
    validation = './data/validation'

    data = []
    for r in [train, validation]:
        classes = glob.glob(r + '/*')
        for cls in tqdm(classes):
            imgs = glob.glob(cls + '/*')
            raws = []

            for img in [imgs[0]]: # use only the first image in each class
                raw = io.imread(img, as_gray=True)
                raw = transform.resize(raw, (28, 28))
                for dg in [0, 90, 180, 270]: # augmentation
                    raw_rot = transform.rotate(raw, dg)
                    raw_rot = raw_rot[:, :, np.newaxis] # (28, 28, 1)
                    raw_rot = raw_rot.astype(np.float32) / 255.
                    raws.append(raw_rot)
            data.append(np.asarray(raws))
    np.save('whale_flukes.npy', np.asarray(data))

class Data_loader():

    def __init__(self, batch_size, n_way=5, k_shot=1, train_mode=True):
        if not os.path.exists('whale_flukes.npy'):
           read_whale_flukes()

        self.batch_size = batch_size
        self.n_way = n_way  # 5 or 20, how many classes the model has to select from
        self.k_shot = k_shot # 1 or 5, how many times the model sees the example

        whale_flukes = np.load('whale_flukes.npy')

        # print(whale_flukes.max())

        np.random.shuffle(whale_flukes)
        assert whale_flukes.dtype == np.float32
        # assert whale_flukes.max() == 1.0
        # assert whale_flukes.min() == 0.0

        if train_mode:
            self.images = whale_flukes[:2975, :4, :, :, :]
            self.num_classes = self.images.shape[0]
            self.num_samples = self.images.shape[1]
        else:
            self.images = whale_flukes[2975:, : 4, :, :, :]
            self.num_classes = self.images.shape[0]
            self.num_samples = self.images.shape[1]

        self.iters = self.num_classes

    def next_batch(self):
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []
        for _ in range(self.batch_size):
            x_set = []
            y_set = []
            x = []
            y = []
            classes = np.random.permutation(self.num_classes)[:self.n_way]
            target_class = np.random.randint(self.n_way)
            for i, c in enumerate(classes):
                samples = np.random.permutation(self.num_samples)[:self.k_shot+1]
                for s in samples[:-1]:
                    x_set.append(self.images[c][s])
                    y_set.append(i)

                if i == target_class:
                    x_hat_batch.append(self.images[c][samples[-1]])
                    y_hat_batch.append(i)

            x_set_batch.append(x_set)
            y_set_batch.append(y_set)

        return np.asarray(x_set_batch).astype(np.float32), np.asarray(y_set_batch).astype(np.int32), np.asarray(x_hat_batch).astype(np.float32), np.asarray(y_hat_batch).astype(np.int32)
