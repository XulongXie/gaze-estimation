import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
import itertools

from Step3.dataloader import get_train_loader as loader
from Step3.dataloader import GazeMe
from Step2.models.model import Network

from collections import OrderedDict
from collections import namedtuple
from itertools import product

from IPython.display import display, clear_output
import pandas as pd
import time
import json
import os

import cv2
import numpy as np
import PIL
from PIL import Image

# 找出准确预测类的个数，可以大约判断这个model的准确性和性能
def get_num_correct(preds, labels):
    # .sum()将所有的元素相加，因为是bool值所以加起来的数就是个数，只是数据结构为tensor
    # .item()取出单元素张量的元素值并返回该值，保持原元素类型不变
    return preds.argmax(dim=1).eq(labels).sum().item()

# 计算均值和标准差
def norm():
    # 此时的batch不再是所有的数据
    train_set, Dataloader = loader('C:/Users/hasee/Desktop/Master_Project/Step1/Label', batch_size = 100, num_workers = 1, is_shuffle = True)
    # 计算像素点的个数
    num_of_pixels = len(train_set) * 224 * 224
    # 计算均值
    total_sum = 0
    # 分批次求和
    print("begin to calculate!")
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
    print(mean, std)
    print("end calculate!")
    return mean, std

# 将训练的数据集做一个标准化
def get_train_set(path, part):
    trans_train = transforms.Compose([
        transforms.Grayscale(1),
        # this also convert pixel value from [0,255] to [0,1]
        transforms.ToTensor(),
        transforms.Normalize(mean = 0.4638,
                             std = 0.1889),
    ])
    train_set = GazeMe(path, part, transform = trans_train)
    print("dataset successfully read!")
    return train_set


class runBuilder():
    # 静态方法，可以不用实例化直接引用
    @staticmethod
    def get_runs(params):
        # namedtuple是一种可读性更强的元组，元组的title是Run，参数类型为字符串序列
        Run = namedtuple('Run', params.keys())
        runs = []
        # v是params里面的values组成的笛卡尔积
        for v in product(*params.values()):
            # 将笛卡尔积的每个值传入而不是当成一个整体，分别传到Run的参数中
            runs.append(Run(*v))
        return runs


class runManager():
    def __init__(self):

        # 记录每个epoch的参数
        self.epoch_count = 0                # epoch的次数
        self.epoch_loss = 0                 # 每次epoch的loss
        self.epoch_num_correct = 0          # 每次epoch正确的个数
        self.epoch_start_time = None        # epoch的起始时间

        # 记录每次运行（不同的超参数背景）
        self.run_params = None              # 超参数的数值
        self.run_count = 0                  # 第几次运行，跟batch_size有关
        self.run_data = []                  # 每次epoch对应的超参数的数值以及计算出的loss等
        self.run_start_time = None          # 每次运行的起始时间

        self.network = None                 # 网络
        self.loader = None                  # 数据
        self.tb = None                      # tensorboard的写入

    # 每次运行开始需要进行的操作，需要传入一个网络和数据以及必要的超参数，放在RunBilder里面管理
    def begin_run(self, run, network, loader):
        # 起始时间
        self.run_start_time = time.time()
        # 记录此次运行的超参数
        self.run_params = run
        # 记录运行的次数
        self.run_count += 1
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment = f'-{run}')
        # 写在tensorboard里面
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        self.tb.add_image('images', grid)
        self.tb.add_graph(
            self.network
            , images.to(getattr(run, 'device', 'cpu'))
        )

    # 每次运行结束时需要进行的操作
    def end_run(self):
        # 关闭tensorboard的写操作
        self.tb.close()
        # 将epoch的次数重新归零
        self.epoch_count = 0
        # 每次epoch开始时需要进行的操作

    def begin_epoch(self):
        # 记录起始时间
        self.epoch_start_time = time.time()
        # 记录epoch的次数
        self.epoch_count += 1
        # 将epoch的loss重新归零
        self.epoch_loss = 0
        # 将epoch的正确个数重新归零
        self.epoch_num_correct = 0

    # 每次epoch结束时需要进行的操作
    def end_epoch(self):
        # 计算每次epoch完成所用的时间
        epoch_duration = time.time() - self.epoch_start_time
        # 计算每次运行（所有epoch）所用时间，这里需要注意，这里其实是在对epoch的时间经行累加
        run_duration = time.time() - self.run_start_time
        # 计算正确率
        loss = self.epoch_loss
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        # tensorboard写入数据
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        # tensorboard写入数据
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            # self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
        # 将结果用表格的形式可视化，每一次epoch是最小单位，所以应该在这里可视化
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        print('runs: ' + "%d" % results["run"] + ', ' + 'epoch: ' + "%d" % results["epoch"] + ', ' +
              'loss: ' + "%d" % results["loss"] + ', ' + 'accuracy: ' + "%f" % results["accuracy"])
        '''
        df = pd.DataFrame.from_dict(self.run_data, orient = 'columns')
        clear_output(wait=True)
        display(df)
        '''


    # 计算loss的方法，batch[0].shape[0]其实就是batch_size
    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    # 计算正确个数的方法的方法
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim = 1).eq(labels).sum().item()

    # 将结果（表格）分别存为excel.csv和json格式
    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data
            , orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding = 'utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii = False, indent = 4)


def train_process(run, net, epoch1, epoch2, optimizer, train_loader):

    # 实例化一个RunManager
    m = runManager()
    print("runManager successfully established!")
    # 开始一次训练，此次训练的超参由传入的run决定
    m.begin_run(run, net, train_loader)
    print("start to run!")
    epoch_num = 1
    for epoch in range(epoch1, epoch2):
        print(epoch_num)
        print("---------------------------------------------------")
        # 每次epoch的开始
        m.begin_epoch()
        # net.train()
        for batch in train_loader:
            # 跟之前一样的操作
            images, labels = batch
            preds = net(images)                    # Pass Batch
            loss = F.cross_entropy(preds, labels)  # Calculate Loss
            optimizer.zero_grad()                  # Zero Gradients
            loss.backward()                        # Calculate Gradients
            optimizer.step()                       # Update Weights
            # 开始计算loss和正确的个数
            m.track_loss(loss, batch)
            m.track_num_correct(preds, labels)
        # 结束一次epoch
        m.end_epoch()
        lr_scheduler.step()
        epoch_num += 1
    # 结束一次运行
    m.end_run()
    # 将所有的结果进行一次保存
    m.save('analysis/results')
    print('Saving state, iter:', str(epoch + 1))
    state = {'epoch': epoch + 1,
                 'model_state': network.state_dict(),
                 'optim_state': optimizer.state_dict(),
                 'scheule_state': lr_scheduler.state_dict()
                 }
    torch.save(state, 'logs/Epoch%d.pth'%(epoch + 1))

if __name__ == '__main__':
    # 创建一个字典存放不同的超参数
    parameters = dict(
        batch_size = [16]
        , num_workers = [2]
        , shuffle = [True]
    )
    # 是否使用Cuda
    Cuda = False
    # 是否使用余弦退火预热
    Cosine_lr = True
    # 输入的shape大小
    input_shape = (224, 224)
    # classes的路径，非常重要
    classes_path = 'C:/Users/hasee/Desktop/Master_Project/Step1/Label'
    # 加载模型
    network = Network()
    # 加载权值，可以用于继续训练
    model_path = "logs/Epoch15.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = network.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    network.load_state_dict(pretrained_dict['model_state'], strict = True)
    print('Finished!')
    # 可选项，有人指出在存在batch normalization的网络里加上这个效果会更好
    net = network.train()
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始epoch
    #   Freeze_Epoch为冻结训练的epoch
    #   Epoch总训练epoch
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#

    if True:
        initial_lr = 1e-3
        Init_Epoch = 15
        Freeze_Epoch = 20
        optimizer = optim.SGD(net.parameters(), lr = initial_lr)
        if Cosine_lr:
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5)
        else:
            lr_scheduler = StepLR(optimizer, step_size = 1, gamma = 0.92)

        # 取出放在run中的超参数
        runs = runBuilder.get_runs(parameters)
        print(runs)
        train_set = get_train_set(classes_path, 'train')
        train_loader = DataLoader(train_set, batch_size = runs[0].batch_size, num_workers = runs[0].num_workers,
                                                   shuffle = runs[0].shuffle)
        print("data successfully loaded!")
            
        # 冻结一定部分训练
        print("start to freeze the backbone!")
        for param in network.backbone.parameters():
            param.requires_grad = False
        print("backbone successfully freezed!")
        print("start to get train!")
        train_process(runs[0], net, Init_Epoch, Freeze_Epoch, optimizer, train_loader)

'''
    if True:
        initial_lr = 1e-4
        Freeze_Epoch = 20
        Unfreeze_Epoch = 25
        optimizer = optim.SGD(net.parameters(), lr = initial_lr)
        if Cosine_lr:
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5)
        else:
            lr_scheduler = StepLR(optimizer, step_size = 1, gamma = 0.92)

        # 取出放在run中的超参数
        runs = runBuilder.get_runs(parameters)
        train_set = get_train_set(classes_path, 'train')
        train_loader = DataLoader(train_set, batch_size = runs[0].batch_size, num_workers=runs[0].num_workers,
                                                   shuffle = runs[0].shuffle)
        print("data successfully loaded!")

        # 解冻后训练
        print("start to unfreeze the backbone!")
        for param in network.backbone.parameters():
            param.requires_grad = True
        print("backbone successfully unfreezed!")
        print("start to get train!")
        train_process(runs[0], net, Freeze_Epoch, Unfreeze_Epoch, optimizer, train_loader)
'''

