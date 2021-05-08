import cv2
import numpy as np
import math

# gamma operation
def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # build the table
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # transfer to numpy.array type
    return cv2.LUT(img, gamma_table)  # use look-up-table

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
    print(gray.shape)
    return gray

# binarization
def Binarization (img):
    # initialization
    img_new = img.astype(np.float32)
    S_N = 0
    P_N = 0
    for i in range(1, img_new.shape[0] - 1):
        for j in range(1, img_new.shape[1] - 1):
            S_N += max(abs(img_new[i - 1, j] - img_new[i + 1, j]), abs(img_new[i, j - 1] - img_new[i, j + 1]))
            P_N += img_new[i, j] * max(abs(img_new[i - 1, j] - img_new[i + 1, j]), abs(img_new[i, j -1] - img_new[i, j + 1]))
    T = P_N / S_N
    T = int(T)
    # binarization
    ret, thresh1 = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, T, 255, cv2.THRESH_BINARY_INV)

    return  thresh1, thresh2

# black hat operation
def blackHat (img, kernel_size):
    # set the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # apply top-hat function
    black = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    return black

# top hat operation
def topHat (img, kernel_size):
    # set the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # apply top-hat function
    top = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

    return top

# resize image without distortion
def letter_box(img, size):
    h = size[0]  # 224
    w = size[1]
    iw, ih = img.shape    # 380, 384
    scale = min(h / ih, w / iw)
    # scale the picture without distortion, if the original picture is not square, it certainly does not meet 224 * 224
    nw = int(iw * scale)
    nh = int(ih * scale)
    mask = np.zeros((size[0], size[1]), dtype = np.uint8)
    new_image = cv2.resize(img, (nw, nh), cv2.INTER_CUBIC)
    for row in range(nh):
        for col in range(nw):
            mask[row, col] = new_image[row, col]
    return mask

if __name__ == '__main__':

    file_path = 'C:/Users/hasee/Desktop/Master_Project/testPic/Me.jpg'
    # original
    img = cv2.imread(file_path)
    # Gray value image
    img_gray = GrayImg(img)
    # light move
    img_Remove = lightRemove(img_gray)
    # calculate the mean value
    mean = np.mean(img_gray)
    # adaptive gamma
    gamma_val = math.log10(0.5) / math.log10(mean / 255)
    # gamma transfer
    image_gamma = gamma_trans(img, gamma_val)
    # back to gray-level
    image_gamma_correct = GrayImg(image_gamma)
    # light move
    img_gamma_Remove = lightRemove(image_gamma_correct)

    cv2.imshow('image_raw', img)
    cv2.imshow('image_gray', img_gray)
    cv2.imshow('image_gamma', image_gamma)
    cv2.imshow('image_gamma_gray', image_gamma_correct)
    cv2.imshow('image_Remove_1', img_gamma_Remove)
    cv2.imshow('image_Remove_2', img_Remove)
    cv2.waitKey(0)

