import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from collections import OrderedDict
from collections import namedtuple
from itertools import product

from IPython.display import display, clear_output
import pandas as pd
import time
import json

torch.set_printoptions(linewidth=120) # Display options for output
torch.set_grad_enabled(True) # Already on by default

from torch.utils.tensorboard import SummaryWriter

# 找出准确预测类的个数，可以大约判断这个model的准确性和性能
def get_num_correct(preds, labels):
    # .sum()将所有的元素相加，因为是bool值所以加起来的数就是个数，只是数据结构为tensor
    # .item()取出单元素张量的元素值并返回该值，保持原元素类型不变
    return preds.argmax(dim=1).eq(labels).sum().item()


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
        self.epoch_count = 0  # epoch的次数
        self.epoch_loss = 0  # 每次epoch的loss
        self.epoch_num_correct = 0  # 每次epoch正确的个数
        self.epoch_start_time = None  # epoch的起始时间

        # 记录每次运行（不同的超参数背景）
        self.run_params = None  # 超参数的数值
        self.run_count = 0  # 第几次运行
        self.run_data = []  # 每次epoch对应的超参数的数值以及计算出的loss等
        self.run_start_time = None  # 每次运行的起始时间

        self.network = None  # 网络
        self.loader = None  # 数据
        self.tb = None  # tensorboard的写入

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
        self.tb = SummaryWriter(comment=f'-{run}')

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
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

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

        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        clear_output(wait=True)
        display(df)

    # 计算loss的方法，batch[0].shape[0]其实就是batch_size
    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    # 计算正确个数的方法的方法
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    # 将结果（表格）分别存为excel.csv和json格式
    def save(self, fileName):

        pd.DataFrame.from_dict(
            self.run_data
            , orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)