from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import nms


# 将网络输出地结果解码为bonding box地形式
class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        # -----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
        #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
        #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
        # -----------------------------------------------------------#
        self.anchors = anchors
        self.num_anchors = len(anchors)   # 3
        self.num_classed = num_classes    # class
        self.img_size = img_size
        self.bbox_attrs = 5 + num_classes # 4 + 1 + class

    # 这里地input是指的某一个输出地特征tensor，大小为batch_size * [3 * (4 + 1 + class)] * 13 * 13
    # 这里相当于是在对特征层进行解码
    def forward(self, input):
        # 获得相应地index的大小
        # -----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 255, 13, 13
        #   batch_size, 255, 26, 26
        #   batch_size, 255, 52, 52
        # -----------------------------------------------#
        batch_size = input.size(0)
        input_width = input.size(2)
        input_height = input.size(3)
        # 将输入的特征图reshape，方便提取到[3 * (4 + 1 + class)]这一维的信息
        # -----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 3, 13, 13, 85
        #   batch_size, 3, 26, 26, 85
        #   batch_size, 3, 52, 52, 85
        # -----------------------------------------------#
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
        # 先验框的中心位置的调整参数，否则会导致在训练初期无法收敛
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]
        h = prediction[..., 3]
        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])
        # 计算缩放比，将设置的先验框转换到特征层的尺度上
        # 例如输入为416x416时，stride_h = stride_w = 32、16、8
        stride_w = self.img_size[0] / input_width
        stride_h = self.img_size[1] / input_height
        # 获得相应特征层的anchor，从输入图片的尺度转换到特征图的尺度
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                          self.anchors]
        # 遍历每一个特征层的点，计算出先验框并放在各个矩阵中，先验框包含了中心点坐标和长宽
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(torch.FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(torch.FloatTensor)
        anchor_w = torch.FloatTensor(scaled_anchors).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.FloatTensor(scaled_anchors).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
        # 利用从特征层中得到的结果对先验框进行调整
        # 首先调整先验框的中心，从先验框中心向右下角偏移，再调整先验框的宽高
        pred_boxes = torch.FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        # 再调整先验框的宽高
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        # 将输出结果调整成相对于输入图像大小
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(torch.FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data

# 不失真的情况下resize图片
def letter_box(img, size):
    w, h = size
    iw, ih = img.size
    scale = min(h/ih, w/iw)
    # 将图片无失真缩放，但此时如果原图不是正方形的话肯定不满足416 * 416
    nh = int(ih * scale)
    nw = int(iw * scale)
    img = img.resize((nw, nh), Image.BICUBIC)
    # 创建一张灰图
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(img, ((w-nw)//2, (h-nh)//2))
    return new_image

# 还原bonding box
def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape/image_shape)

    # 这里的偏移量是归一化后的
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    # 将得到的bonding box的参数整合到一起，重新得到中心点和长宽并归一化
    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)/input_shape
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape
    # 通过偏移和缩放去除有灰条的部分
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    # 重新将中心点和长宽转换成box -> top, left, bottom, right
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    # 得到在原图上的top, left, bottom, right
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

# 计算IOU
def bbox_iou(box1, box2, x1y1x2y2=True):
    # 首先判断输入的box格式是否是左上，右下两个坐标点(x1, y1), (x2, y2), 如果不是，还需要将输入的格式(x, y, w, h)转换成左上右下的格式
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2    # box1的左上右下的横坐标
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2    # box1的左上右下的纵坐标
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2    # box2的左上右下的横坐标
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2    # box1的左上右下的纵坐标
    # 如果已经是，就可以直接用了
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    # 计算出相交部分的左上右下的坐标点
    inter_rect_x1 = torch.max(b1_x1, b2_x1)    # 左上角是比谁大
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)    # 右下角是比谁小
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # 计算出相交部分的面积，用clamp限制面积不可以小于0
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # 计算两个box的面积
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    # 计算交并比，后面要减去重叠的部分
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

# 极大值抑制，对于一个类每次只返回一个框
def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    # 将预测结果的格式转换成左上角右下角的格式
    # prediction  [batch_size, num_anchors, 85] x, y, w, h, obj_conf, class_conf
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    # 计算完了再传回去
    prediction[:, :, :4] = box_corner[:, :, :4]
    # 最后传回去的[]只需要有batch_size的长度，每个位置存放的是85个信息
    output = [None for _ in range(len(prediction))]
    # image_i是batch_size中每一张图片，image_pred是后面两个维度
    for image_i, image_pred in enumerate(prediction):
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类的index
        # ----------------------------------------------------------#
        # 在torch.max()中指定了dim之后，比如对于一个3x4x5的Tensor，指定dim为0后，此时输出的Tensor的维度是4x5
        # 返回的是3个先验框最有可能的class
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        # 利用置信度进行第一轮筛选，选出置信度相对于大一点的类
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
        # 根据置信度进行预测结果的筛选
        image_pred = image_pred[conf_mask]    # 这一步主要是为下面的“7”做准备
        # 这一步是因为置信度最大的类也有可能不满足达到阈值的要求，所以要筛选一下
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        # detections  [num_anchors, 7]
        # 7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # 获得预测结果中包含的所有种类，unique是为了排除重复的，以免for循环重复
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            # 获得某一类得分筛选后全部的预测结果
            detections_class = detections[detections[:, -1] == c]

            # 非极大值抑制，目的是选出某各类中最大的得分
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]

            # 有可能这三个先验框检测出了大于一个的类，所以要拼接
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output