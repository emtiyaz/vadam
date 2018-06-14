import os
import torch
import torch.utils.data as data
import numpy as np
import re

class BostonPresplit(data.Dataset):

    def __init__(self, root, data_set, train=True):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        self.data_folder = os.path.join(self.root, "bostonHousing")
        self.data_file = os.path.join(self.data_folder, "data.txt")
        self.idx_target = os.path.join(self.data_folder, "index_target.txt")
        self.idx_features = os.path.join(self.data_folder, "index_features.txt")
        data = np.loadtxt(self.data_file)
        idx_target = np.loadtxt(self.idx_target, dtype=int).tolist()
        idx_features = np.loadtxt(self.idx_features, dtype=int).tolist()

        x = data[:,idx_features]
        y = data[:,idx_target]
        
        split_num = re.findall("\d+", data_set)[0]

        self.idx_train = os.path.join(self.data_folder, "index_train_" + split_num + ".txt")
        self.idx_test = os.path.join(self.data_folder, "index_test_" + split_num + ".txt")

        idx_train = np.loadtxt(self.idx_train, dtype=int).tolist()
        idx_test = np.loadtxt(self.idx_test, dtype=int).tolist()

        if self.train:
            X_train = x[idx_train]
            y_train = y[idx_train]
            self.train_data, self.train_labels = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        else:
            X_test = x[idx_test]
            y_test = y[idx_test]
            self.test_data, self.test_labels = torch.FloatTensor(X_test), torch.FloatTensor(y_test)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            x, y = self.train_data[index], self.train_labels[index]
        else:
            x, y = self.test_data[index], self.test_labels[index]

        return x, y

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
