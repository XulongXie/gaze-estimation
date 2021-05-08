import torch
from torch.utils.data import DataLoader

from Step2.Plan_B.GazeData import get_train_set, get_val_set

# 计算均值和标准差
def norm():
    # 此时的batch不再是所有的数据
    train_set = get_train_set('C:/Users/hasee/Desktop/Master_Project/Step1/Label')
    val_set = get_val_set('C:/Users/hasee/Desktop/Master_Project/Step1/Label')
    # 加载dataloader
    train_loader = DataLoader(train_set, batch_size = 100, shuffle = True, num_workers = 2)
    val_loader = DataLoader(val_set, batch_size = 100, shuffle = True, num_workers = 2)
    # 计算像素点的个数
    num_of_pixels_train = len(train_set) * 224 * 224
    num_of_pixels_val = len(val_set) * 224 * 224
    # 计算均值
    total_sum_train = 0
    total_sum_val = 0
    # 分批次求和
    print("begin to calculate!")
    for batch in train_loader:
        total_sum_train += batch[0].sum()
    for batch in val_loader:
        total_sum_val += batch[0].sum()
    # 计算均值
    mean_train = total_sum_train / num_of_pixels_train
    mean_val = total_sum_val / num_of_pixels_val
    # 计算标准差
    sum_of_squared_error_train = 0
    sum_of_squared_error_val = 0
    # 分批次求均方差
    for batch in train_loader:
        sum_of_squared_error_train += ((batch[0] - mean_train).pow(2)).sum()
    for batch in val_loader:
        sum_of_squared_error_val += ((batch[0] - mean_val).pow(2)).sum()
    std_train = torch.sqrt(sum_of_squared_error_train / num_of_pixels_train)
    std_val = torch.sqrt(sum_of_squared_error_val / num_of_pixels_val)
    print("end calculate!")
    return mean_train, std_train, mean_val, std_val

if __name__ == '__main__':
    mean_train, std_train, mean_val, std_val = norm()
    print(mean_train, std_train)
    print(mean_val, std_val)