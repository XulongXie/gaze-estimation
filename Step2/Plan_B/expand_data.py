import os
import multiprocessing
import cv2

import numpy as np


# gamma operation
def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # build the table
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # transfer to numpy.array type
    return cv2.LUT(img, gamma_table)  # use look-up-table


# Gaussian blur
def Gaussian_blur(img, kernel_size, std):
    # gaussian blurred image
    gaussian = cv2.GaussianBlur(img, (kernel_size, kernel_size), std)
    return gaussian


# Median filter
def Median_filter(img, kernel_size):
    # median blurred image
    median = cv2.medianBlur(img, kernel_size)
    return median


# Gaussian noise
def gaussian_noise(img, mean, var):
    img = img.astype(np.float32)
    noise = np.random.normal(mean, var, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0.0, 255.0)
    noisy = noisy.astype(np.uint8)
    return noisy


# rotation
def rotation(img, k):
    # rotate 90 ~ 270
    rotate = np.rot90(img, k)
    return rotate


# flip
def flipImg(img):
    return cv2.flip(img, 1)


# expand the dataset
def expand(src, dst):
    # src = 'C:/Users/hasee/Desktop/Master Project/Step2/Plan_B/DataSet/480x270'
    # dst = 'C:/Users/hasee/Desktop/Master Project/Step1/Expand'
    src_list = src.split('/')
    class_name = src_list[-1]  # for example 480x270
    # output_dir = 'C:/Users/hasee/Desktop/Master Project/Step2/Plan_B/Expand/480x270'
    output_dir = dst + '/' + class_name
    # create a new dir if it not exist
    try:
        os.mkdir(output_dir)
        print(output_dir + ' 创建成功')
    except:
        pass
    i = 1
    # get the images
    img_list = os.listdir(src)
    for img in img_list:
        srcPath = src + '/' + img
        src_img = cv2.imread(srcPath)
        # gaussian noise
        noise = gaussian_noise(src_img, 0, 4)
        output_path = os.path.join(output_dir, "%06d.jpg" % i)
        cv2.imwrite(output_path, noise)
        i += 1
        # gaussian blur
        gaussian1 = Gaussian_blur(src_img, 3, 1)
        output_path = os.path.join(output_dir, "%06d.jpg" % i)
        cv2.imwrite(output_path, gaussian1)
        i += 1
        gaussian2 = Gaussian_blur(src_img, 3, 2)
        output_path = os.path.join(output_dir, "%06d.jpg" % i)
        cv2.imwrite(output_path, gaussian2)
        i += 1
        median = Median_filter(src_img, 3)
        output_path = os.path.join(output_dir, "%06d.jpg" % i)
        cv2.imwrite(output_path, median)
        i += 1
        gamma1 = gamma_trans(src_img, 1.2)
        output_path = os.path.join(output_dir, "%06d.jpg" % i)
        cv2.imwrite(output_path, gamma1)
        i += 1
        gamma2 = gamma_trans(src_img, 1 / 1.8)
        output_path = os.path.join(output_dir, "%06d.jpg" % i)
        cv2.imwrite(output_path, gamma2)
        i += 1


# main function
if __name__ == '__main__':
    Path = "C:/Users/hasee/Desktop/Master_Project/Step2/Plan_B/DataSet"
    dstPath = "C:/Users/hasee/Desktop/Master_Project/Step2/Plan_B/Expand"

    # create a new one if it's not existing
    try:
        os.mkdir(dstPath)
        print(dstPath + ' 创建成功')
    except:
        print("文件夹已经存在，无需创建")

    # read all kinds of folders in the data set folder
    dir_list = os.listdir(Path)
    # stitch the names of each folder
    for dir_name in dir_list:
        # find the folders of each category
        srcPath = Path + "/" + dir_name
        # multi-processing process
        lable_process = multiprocessing.Process(target = expand, args = (srcPath, dstPath))
        lable_process.start()
