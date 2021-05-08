import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np

from Step2.preProcessing.PreProcess import letter_box


class GazeMe(data.Dataset):
    # __init__ method will read data file
    def __init__(self, dir_path, part, transform = None):
        # initialize the properties inherited from the parent class
        super(GazeMe, self).__init__()
        # absolute path of all pictures
        assert part in ["train", "test", "val"]
        sample = []
        with open(os.path.join(dir_path, part, "label.txt")) as f:
            for line in f:
                # remove useless spaces at the beginning and end
                line = line.rstrip()
                # separate pictures and tags according to spaces
                words = line.split()
                sample.append((words[0], int(words[1]), int(words[2]), int(words[3]), int(words[4]),
                               int(words[5]), int(words[6]), int(words[7]), int(words[8]), int(words[9])))
                self.sample = sample
                self.transform = transform

    # __len__ method returns the size of the custom data set, which is convenient for later traversal
    def __len__(self):
        # return the length of datasets
        return len(self.sample)

    # __getitem__ method supports subscript access
    def __getitem__(self, index):
        img, leftCorner1x, leftCorner1y, leftCorner2x, leftCorner2y, rightCorner1x, rightCorner1y, \
        rightCorner2x, rightCorner2y, label = self.sample[index]
        img = Image.open(img)
        leftEye = img.crop([leftCorner1x, leftCorner1y, leftCorner2x, leftCorner2y])
        leftEye = leftEye.resize((64, 64))
        rightEye = img.crop([rightCorner1x, rightCorner1y, rightCorner2x, rightCorner2y])
        rightEye = rightEye.resize((64, 64))
        eyeCorners = np.array([leftCorner1x, leftCorner1y, leftCorner2x, leftCorner2y, rightCorner1x,
                               rightCorner1y, rightCorner2x, rightCorner2y])
        eyeCorners = torch.from_numpy(eyeCorners)
        eyeCorners = eyeCorners.reshape(1, 8)
        if self.transform is not None:
            leftEye = self.transform(leftEye)
            rightEye = self.transform(rightEye)
        return leftEye, rightEye, eyeCorners, label

def get_train_set(data_dir):
    trans_train = transforms.Compose([
        transforms.Grayscale(1),
        # this also convert pixel value from [0,255] to [0,1]
        transforms.ToTensor(),
        # 这个还要修改
        transforms.Normalize(mean = 0.4638,
                             std = 0.1889),
    ])
    # load dataset
    # 'C:/Users/hasee/Desktop/Master Project/Step1/Label'
    train_set = GazeMe(dir_path = data_dir, part = "train", transform = trans_train)
    print("train set successfully read!")
    return train_set


def get_val_set(data_dir):
    trans_train = transforms.Compose([
        transforms.Grayscale(1),
        # this also convert pixel value from [0,255] to [0,1]
        transforms.ToTensor(),
        # 这个还要修改
        transforms.Normalize(mean = 0.4638,
                             std = 0.1889),
    ])
    # load dataset
    # 'C:/Users/hasee/Desktop/Master Project/Step1/Label'
    val_set = GazeMe(dir_path = data_dir, part = "test", transform = trans_train)
    print("validation set successfully read!")
    return val_set