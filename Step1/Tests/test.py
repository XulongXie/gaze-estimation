import os
import numpy as np
import  cv2
import multiprocessing
from PIL import Image

# Gaussian noise
def gaussian_noise(img, mean, var):
    shape = img.shape
    img = img.astype(np.float32)
    noise = np.random.normal(mean, var, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0.0, 255.0)
    noisy = noisy.astype(np.uint8)
    return noisy

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

# rotation
def rotation(img, k):
    # rotate 90 ~ 270
    rotate = np.rot90(img, k)
    return rotate

# flip
def flipImg(img):
    return cv2.flip(img, 1)

# gamma operation
def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # build the table
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)          # transfer to numpy.array type
    return cv2.LUT(img, gamma_table)

# Gray-Level
def GrayImg (img):
    shape = img.shape
    gray = np.zeros((shape[0], shape[1]), dtype = np.float32)
    # loop for every pixels
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            # attention: G,B,R
            # I(x,y)) = 0.7152G + 0.0722B + 0.2126R
            gray[i, j] = 0.7152 * img[i, j, 0]  + 0.0722 * img[i, j, 1] + 0.2126 * img[i, j, 2]
    # convert to uint8
    gray = gray.astype(np.uint8)
    return gray

# calculate the threshold
def CalculateM (img):
    # calculate the row
    row = img.shape[0]
    upper_row = row // 2
    pixel_list = []

    for i in range(0, upper_row):
        for j in range(0, img.shape[1]):
            pixel_list.append(img[i, j])

    # sort the list
    pixel_list.sort()
    # select the median gray-level value
    length = len(pixel_list)
    if (length % 2 != 0):
        index = length // 2
        value = pixel_list[int(index)]
    else:
        index1 = (length - 1) // 2
        index2 = length / 2
        value = int((int(pixel_list[int(index1)]) + int(pixel_list[int(index2)])) / 2)

    return value

# remove the light influence
def lightRemove (img):
    afterRemove = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    # calculate M
    M = CalculateM(img)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if (img[i, j] <= M):
                afterRemove[i, j] = img[i, j]
            else:
                afterRemove[i, j] = M

    afterRemove = afterRemove.astype(np.uint8)
    return afterRemove

# 不失真的情况下resize图片
def letter_box(img, size):
    h = size  # 224
    w = size
    iw = img.shape[0]    # 380, 384
    ih = img.shape[1]
    scale = min(h / ih, w / iw)
    # 将图片无失真缩放，但此时如果原图不是正方形的话肯定不满足224 * 224
    nw = int(iw * scale)
    print(nw)
    nh = int(ih * scale)
    print(nh)
    mask = np.zeros((size, size), dtype = np.uint8)
    new_image = cv2.resize(img, (nw, nh), cv2.INTER_CUBIC)
    for row in range(nh):
        for col in range(nw):
            mask[row + (h - nh) // 2, col + (w - nw)//2] = new_image[row, col]
    return mask

def recreate(src, dst):
    # src = 'C:/Users/hasee/Desktop/Master Project/Step1/DataSet/300x150'
    # dst = 'C:/Users/hasee/Desktop/data'
    src_list = src.split('/')
    class_name = src_list[-1]  # for example 300x150
    # output_dir = 'C:/Users/hasee/Desktop/DataSet/300x150'
    output_dir = dst + '/' + class_name
    # create a new dir if it not exist
    try:
        os.mkdir(output_dir)
        print(output_dir + ' 创建成功')
    except:
        pass
    # get the images
    img_list = os.listdir(src)
    for img in img_list:
        srcPath = src + '/' + img
        output_path = output_dir + '/' + img
        src_img = cv2.imread(srcPath, cv2.IMREAD_UNCHANGED)
        out_img = letter_box(src_img, 224)
        cv2.imwrite(output_path, out_img)

# 主函数
if __name__ == '__main__':
    '''
    Path = "C:/Users/hasee/Desktop/Master_Project/Step1/DataSet"
    dstPath = "C:/Users/hasee/Desktop/data"

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
        recreate_process = multiprocessing.Process(target = recreate, args = (srcPath, dstPath))
        recreate_process.start()
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi, 100)
    y1, y2 = np.sin(x), np.cos(x)

    plt.plot(x, y1)
    plt.plot(x, y2)

    plt.title('line chart')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()



