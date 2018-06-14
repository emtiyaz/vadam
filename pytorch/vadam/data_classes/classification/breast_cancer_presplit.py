import os
import torch
import torch.utils.data as data
import numpy as np

class BreastCancerPresplit(data.Dataset):

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        if self.train:
            self.training_file_x = os.path.join(self.root, "breast_cancer_scale_presplit/breast_cancer_scale_X_tr.csv")
            self.training_file_y = os.path.join(self.root, "breast_cancer_scale_presplit/breast_cancer_scale_y_tr.csv")
            X_train = np.loadtxt(self.training_file_x)
            y_train = np.loadtxt(self.training_file_y)
            self.train_data, self.train_labels = torch.FloatTensor(X_train), torch.LongTensor(y_train)
        else:
            self.test_file_x = os.path.join(self.root, "breast_cancer_scale_presplit/breast_cancer_scale_X_te.csv")
            self.test_file_y = os.path.join(self.root, "breast_cancer_scale_presplit/breast_cancer_scale_y_te.csv")
            X_test = np.loadtxt(self.test_file_x)
            y_test = np.loadtxt(self.test_file_y)
            self.test_data, self.test_labels = torch.FloatTensor(X_test), torch.LongTensor(y_test)

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
