import  os
import cv2
import  multiprocessing

# 拷贝函数
def copy(src, dst, name):
    # 首先需要找到目标的文件，所以要进行拼接
    src_file = src + "/" + name
    dst_file = dst + "/" + name
    # 读取图片列表
    img_list = os.listdir(src_file)
    for img in img_list:
        src_Path = src_file + '/' + img
        dst_Path = dst_file + '/' + img
        src_img = cv2.imread(src_Path)
        cv2.imwrite(dst_Path, src_img)



# 主函数
if __name__ == '__main__':
    srcPath = "C:/Users/hasee/Desktop/Master_Project/Step1/Expand"
    dstPath = "C:/Users/hasee/Desktop/Master_Project/Step1/DataSet"

    # 如果没有就创建一个
    try:
        os.mkdir(dstPath)
    except:
        print("文件夹已经存在，无需创建")

    #读取源文件夹中的文件
    file_list = os.listdir(srcPath)
    # 调用拷贝函数
    for file_name in file_list:
        # 使用多进程，也就是拷贝第一个的同时，其他文件也在拷贝
        copy_process = multiprocessing.Process(target = copy, args = (srcPath, dstPath, file_name))
        copy_process.start()
