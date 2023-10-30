"""

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
###############

import os
import os.path
import numpy as np
import sys
import torch

from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torch.utils.data import Dataset as VisionDataset
from utils import check_integrity, download_and_extract_archive

import shutil

class CIFAR10_2():
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):


        self.url = "https://www.dropbox.com/s/ow0wldxbxmqmtzz/cifar10_trainval.tar.gz?dl=1"
        self.url2 = "https://www.dropbox.com/s/1tk8nv0b57o1lr8/cifar10-batches-images-test.tar.gz?dl=1"
                    
        self.filename = "cifar10_trainval.tar.gz"
        self.filename2 = "cifar10-batches-images-test.tar.gz"
        self.transform = transform
        self.target_transform = target_transform

        self.root = root
        if download:
            self.download()
        self.data = []
        self.targets = []
        self.train = train
        #self.test = test

        #if self.test:
        #    img_name = os.path.join(root, "cifar10-batches-images-test/cifar10-batches-images-test.npy")
        #    self.data = np.load(img_name)
        #else:
        if self.train:
            img_name = os.path.join(root, "cifar10_train_val/cifar10-batches-images-train.npy")
            target_name = os.path.join(root, "cifar10_train_val/cifar10-batches-labels-train.npy")
        else:
            #img_name = os.path.join(root, "cifar10_train_val/cifar10-batches-images-val.npy")
            img_name = os.path.join(root, "cifar10-batches-images-test/cifar10-batches-images-test.npy")
            target_name = os.path.join(root, "cifar10_train_val/cifar10-batches-labels-val.npy")

        self.data = np.load(img_name)
        self.targets = np.load(target_name)



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


    def download(self):
        try:
            download_and_extract_archive(self.url, self.root, filename=self.filename)
            download_and_extract_archive(self.url2, self.root, filename=self.filename2)
        except Exception as e:
            print("Interrupted during dataset downloading. "
                  "Cleaning up...")
            # Clean up
            cwd = os.getcwd()
            rm_path = os.path.join(cwd, self.root, "cifar10_trainval")
            shutil.rmtree(rm_path)
            raise e

        print('Files already downloaded and verified')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

