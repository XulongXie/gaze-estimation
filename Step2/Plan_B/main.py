import shutil, os, time, argparse
from collections import OrderedDict
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from Step2.Plan_B.GazeNet import GazeNetModel as Network
from Step2.Plan_B.GazeData import get_train_set, get_val_set

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR


# return a boolean value for usage
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--sink', type = str2bool, nargs = '?', const = True, default = False, help = "Just sink and terminate.")
parser.add_argument('--reset', type = str2bool, nargs = '?', const = True, default = False,
                    help = "Start from scratch (do not load).")
args = parser.parse_args()

# Change there flags to control what happens.
doLoad = False  # Load checkpoint at the beginning
doTest = False  # Only run test, no training

workers = 4
epochs = 100
# Change if out of cuda memory
batch_size = 16

initial_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
lr = initial_lr

Cosine_lr = True

count_test = 0
count = 0
best_loss = 1e20

class runManager():
    def __init__(self):

        # 记录每个epoch的参数
        self.epoch_count = 0                # epoch的次数
        self.epoch_loss = 0                 # 每次epoch的loss
        self.epoch_num_correct = 0          # 每次epoch正确的个数
        self.epoch_start_time = None        # epoch的起始时间

        # 记录每次运行（不同的超参数背景）
        self.run_count = 0                  # 第几次运行，跟batch_size有关
        self.run_start_time = None          # 每次运行的起始时间
        self.loader = None                  # 数据

    # 每次运行开始需要进行的操作，需要传入一个网络和数据以及必要的超参数，放在RunBilder里面管理
    def begin_run(self, loader):
        # 起始时间
        self.run_start_time = time.time()
        # 记录运行的次数
        self.run_count += 1
        self.loader = loader

    # 每次运行结束时需要进行的操作
    def end_run(self):
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
        # 将结果用表格的形式可视化，每一次epoch是最小单位，所以应该在这里可视化
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count + 1
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        print('runs: ' + "%d" % results["run"] + ', ' + 'epoch: ' + "%d" % results["epoch"] + ', ' +
              'loss: ' + "%d" % results["loss"] + ', ' + 'accuracy: ' + "%f" % results["accuracy"])

    # 计算loss的方法，batch[0].shape[0]其实就是batch_size
    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    # 计算正确个数的方法的方法
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim = 1).eq(labels).sum().item()


def main():
    global args, best_loss, weight_decay, momentum

    net = Network()

    epoch = 0
    saved = load_checkpoint()

    # 这里还需要做修改
    dataTrain = get_train_set(data_dir = 'C:/Users/hasee/Desktop/Master_Project/Step2/Plan_B/Label')
    dataVal = get_val_set(data_dir = 'C:/Users/hasee/Desktop/Master_Project/Step2/Plan_B/Label')

    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size = batch_size, shuffle = True,
        num_workers = workers, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size = batch_size, shuffle = False,
        num_workers = workers, pin_memory = True)

    # 效果如果还可以的话可以考虑去掉weight_decay再试试看
    optimizer = torch.optim.SGD(net.parameters(), lr,
                                momentum = momentum,
                                weight_decay = weight_decay)
    # 余弦退火
    if Cosine_lr:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5)
    else:
        lr_scheduler = StepLR(optimizer, step_size = 1, gamma = 0.92)

    if doLoad:
        if saved:
            print('Loading checkpoint for epoch %05d ...' % (saved['epoch']))
            state = saved['model_state']
            try:
                net.module.load_state_dict(state)
            except:
                net.load_state_dict(state)
            epoch = saved['epoch']
            best_loss = saved['best_loss']
            optimizer = saved['optim_state']
            lr_scheduler = saved['scheule_state']
        else:
            print('Warning: Could not read checkpoint!')

    # Quick test
    if doTest:
        validate(val_loader, net, epoch)
        return

    '''
    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
    '''

    m.begin_run(train_loader)
    print("start to run!")
    for epoch in range(epoch, epochs):

        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, net, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        loss_total = validate(val_loader, net, epoch)

        # remember best loss and save checkpoint
        is_best = loss_total < best_loss
        best_loss = min(loss_total, best_loss)
        state = {'epoch': epoch + 1,
                 'model_state': net.state_dict(),
                 'optim_state': optimizer.state_dict(),
                 'scheule_state': lr_scheduler.state_dict(),
                 'best_loss': best_loss,
                 }
        save_checkpoint(state, is_best)
    # 结束一次运行
    m.end_run()


def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()
    m.begin_epoch()
    for i, batch in enumerate(train_loader):
        imEyeL, imEyeR, EyeCorner, gaze = batch

        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = True)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = True)
        EyeCorner = torch.autograd.Variable(EyeCorner.float(), requires_grad = True)
        gaze = torch.autograd.Variable(gaze, requires_grad=False)


        # compute output
        preds = model(imEyeL, imEyeR, EyeCorner)
        # compute loss
        loss = F.cross_entropy(preds, gaze)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算loss和正确的个数
        m.track_loss(loss, batch)
        m.track_num_correct(preds, gaze)

        print('Epoch (train): [{0}][{1}/{2}]\t'.format(epoch + 1, i, len(train_loader)))
    # 结束一次epoch
    m.end_epoch()


def validate(val_loader, model, epoch):
    # switch to evaluate mode
    batch_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()
    model.eval()
    end = time.time()
    for i, batch in enumerate(val_loader):
        imEyeL, imEyeR, EyeCorner, gaze = batch

        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
        EyeCorner = torch.autograd.Variable(EyeCorner.float(), requires_grad = False)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)


        # compute output
        with torch.no_grad():
            preds = model(imEyeL, imEyeR, EyeCorner)

        loss = F.cross_entropy(preds, gaze)
        preds = preds.argmax(dim = 1)
        lossLin = preds - gaze
        lossLin = torch.mul(lossLin, lossLin)
        lossLin = torch.sum(lossLin, 0)
        # lossLin = torch.mean(torch.sqrt(torch.tensor(lossLin,dtype = torch.float)))
        lossLin = torch.mean(torch.sqrt(lossLin.float()))

        losses.update(loss.data.item(), imEyeL.size(0))
        lossesLin.update(lossLin.item(), imEyeL.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch (val): [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
            epoch, i, len(val_loader), batch_time = batch_time,
            loss = losses, lossLin = lossesLin))

    return loss


# 这个可以根据自己需要的路径来改
CHECKPOINTS_PATH = '.'

def load_checkpoint(filename = 'checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state


def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


# Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 自己的训练策略
def adjust_learning_rate(optimizer, epoch):
    # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    # 实例化一个RunManager
    m = runManager()
    main()
    print('DONE')
