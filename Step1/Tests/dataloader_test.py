import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt


class GazeMe(Dataset):
    # __init__ method will read data file
    def __init__(self, dir_path, part, transform=None):
        # initialize the properties inherited from the parent class
        super(GazeMe, self).__init__()
        # absolute path of all pictures
        assert part in ["train", "val"]
        sample = []
        with open(os.path.join(dir_path, part, "test.txt")) as f:
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
        #leftEye = leftEye.resize((64, 64))
        rightEye = img.crop([rightCorner1x, rightCorner1y, rightCorner2x, rightCorner2y])
        #rightEye = rightEye.resize((64, 64))
        eyeCorners = np.array([leftCorner1x, leftCorner1y, leftCorner2x, leftCorner2y, rightCorner1x,
                               rightCorner1y, rightCorner2x, rightCorner2y])
        eyeCorners = torch.from_numpy(eyeCorners)
        eyeCorners = eyeCorners.reshape(1, 8)
        if self.transform is not None:
            leftEye = self.transform(leftEye)
            rightEye = self.transform(rightEye)
        return leftEye, rightEye, eyeCorners, label


def get_train_loader(data_dir, batch_size, num_workers, is_shuffle):
    # load dataset
    # 'C:/Users/hasee/Desktop/Master Project/Step1/Label'
    train_set = GazeMe(dir_path=data_dir, part="train",
                       transform=transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()]))
    # create a dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=is_shuffle)

    return train_set, train_loader


def PIL_to_tensor(image):
    loader = transforms.Compose([
        transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image


def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    # remove the fake batch dimension
    image = image.squeeze(0)
    image = unloader(image)
    return image


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    # we clone the tensor to not do changes on it
    image = tensor.cpu().clone()
    # remove the fake batch dimension
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    # pause a bit so that plots are updated
    plt.pause(0.001)


if __name__ == '__main__':
    '''
    train_set, train_loader = get_train_loader('C:/Users/hasee/Desktop/Master_Project/Step1/Label', 10, 1, True)
    print(len(train_set))
    sample = next(iter(train_set))
    print(type(sample))
    print(len(sample))
    image, label = sample
    print(type(image))
    print(image.shape)
    print(label)
    print(type(label))
    image = tensor_to_PIL(image)
    image.show()

    transform = transforms.Compose([transforms.ToTensor()])
    img = Image.open("../testPic/Me.jpg")
    print(type(img))
    print(img.size)
    img = transform(img)
    print(type(img))
    print(img.shape)
    '''
    train_set, train_loader = get_train_loader('C:/Users/hasee/Desktop/Master_Project/Step1/Label', batch_size=100,
                                             num_workers=1,
                                             is_shuffle=True)
    print(len(train_set))
    sample = next(iter(train_set))
    print(type(sample))
    print(len(sample))
    leftEye, rightEye, eyeCorners, label = sample
    print(eyeCorners)

    image = tensor_to_PIL(rightEye)
    image.show()


