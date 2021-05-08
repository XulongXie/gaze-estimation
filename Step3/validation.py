import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from Step3.dataloader import GazeMe
from Step2.models.model import Network

import os


class gazeNet_eval(object):
    _defaults = {
        # 这里还需要修改
        "model_path": 'model_data/yolo4_weights.pth',
        # 有CUDA的话需要设为true
        "cuda": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # 初始化网络
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)


    # 生成模型
    def generate(self):
        # 建立网络
        self.network = Network().eval()

        # 载入网络的权重
        print('Loading weights into state dict...')
        # 留下接口方便以后用GPU训练
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location = device)
        self.network.load_state_dict(state_dict)
        print('Finished!')

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.network = nn.DataParallel(self.network)
            self.network = self.network.cuda()

        return self.network



# 找出准确预测类的个数，可以大约判断这个model的准确性和性能
def get_num_correct(preds, labels):
    # .sum()将所有的元素相加，因为是bool值所以加起来的数就是个数，只是数据结构为tensor
    # .item()取出单元素张量的元素值并返回该值，保持原元素类型不变
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_data_loader(data_dir, part, batch_size, num_workers, is_shuffle):
    # load dataset
    # 'C:/Users/hasee/Desktop/Master Project/Step1/Label'
    data_set = GazeMe(dir_path = data_dir, part = part, transform = transforms.Compose([transforms.ToTensor()]))
    # create a dataloader
    data_loader = DataLoader(data_set, batch_size = batch_size, num_workers = num_workers, shuffle = is_shuffle)

    return data_set, data_loader

# 计算均值和标准差
def norm(path, part):
    # 此时的batch不再是所有的数据
    data_set, Dataloader = get_data_loader(path, part, 100, 4, True)
    # 计算像素点的个数
    num_of_pixels = len(data_set) * 224 * 224
    # 计算均值
    total_sum = 0
    # 分批次求和
    for batch in Dataloader:
        total_sum += batch[0].sum()
    # 计算均值
    mean = total_sum / num_of_pixels
    # 计算标准差
    sum_of_squared_error = 0
    # 分批次求均方差
    for batch in Dataloader:
        sum_of_squared_error += ((batch[0] - mean).pow(2)).sum()
    std = torch.sqrt(sum_of_squared_error / num_of_pixels)
    return mean, std


# 将测试的数据集做一个标准化
def get_data_set(path, part):
    mean, std = norm(path)
    trans = transforms.Compose([
        # this also convert pixel value from [0,255] to [0,1]
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std),
    ])
    data_set = GazeMe(path, part, transform = trans)
    return data_set

# 定义一个预测的函数
def prediction(modul, loader):
    # 创立一个张量存储所有的预测结果
    preds_set = torch.tensor([])
    for batch in loader:
        images, labels = batch
        # 开始预测
        preds = modul(images)
        # 将所有的预测结果合并到一个张量中
        preds_set = torch.cat((preds_set, preds), dim = 0)
    #返回结果
    return preds_set


def val_process(net):
    pass

# 画出混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    # 是否使用Cuda
    Cuda = False
    # 验证集的路径，非常重要
    train_path = 'C:/Users/hasee/Desktop/Master_Project/Step1/Label'
    test_path = 'C:/Users/hasee/Desktop/Master_Project/Step1/Label'
    # 加载模型
    gaze_net = gazeNet_eval()
    network = gaze_net.generate()
    # 加载训练集，测试经验误差
    train_set = get_data_set(train_path, 'train')
    # 加载测试集，测试泛化误差
    val_set = get_data_set(test_path, 'val')
    # 在预测时不需要梯度跟踪，可以局部将其关掉，这个很有用
    with torch.no_grad():
        # 这里重新创立一个train_loader，首先测试经验误差
        prediction_loader = torch.utils.data.DataLoader(train_set, batch_size = 100)
        # 预测结果
        preds_set = prediction(network, prediction_loader)
    # 计算总的正确的个数
    correct_num = get_num_correct(preds_set, train_set.targets)
    print("正确率个数为：", correct_num)
    print("正确率为：", format(correct_num / len(train_set), '.2%'))
    # 直接利用第三方库函数算出混淆矩阵，左边为真实值，右边为预测值
    Confusion_Matrix = confusion_matrix(train_set.targets, preds_set.argmax(dim=1))
    print(Confusion_Matrix)
    # 画出混淆矩阵
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(Confusion_Matrix, train_set.classes)



