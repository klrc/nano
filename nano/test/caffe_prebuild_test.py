# -*- coding: UTF-8 -*-
from __future__ import division
import math
import caffe
import numpy as np
import cv2
import os
import shutil

INPUT_WIDTH = 416
INPUT_HEIGHT = 224
INPUT_WIDTH_PADDED = INPUT_WIDTH
INPUT_HEIGHT_PADDED = INPUT_HEIGHT

NUM_STRIDES = 3
MAX_WH = 14096  # for nms offset
LABEL_OFFSET = 1


def padimg2(img):
    width, height = img.shape[1], img.shape[0]

    neww = width
    newh = height

    coef = float(width) / height
    coefstd = float(INPUT_WIDTH) / INPUT_HEIGHT
    if coef > coefstd:
        newh = width / coefstd
    else:
        neww = height * coefstd
    neww = int(neww)
    newh = int(newh)

    newimg = np.zeros((newh, neww, 3), np.uint8)
    newimg[:height, :width] = img

    newimg = cv2.resize(newimg, (INPUT_WIDTH, INPUT_HEIGHT))

    global INPUT_WIDTH_PADDED, INPUT_HEIGHT_PADDED
    INPUT_WIDTH_PADDED = neww
    INPUT_HEIGHT_PADDED = newh

    return newimg


# nms算法
def nms(dets, thresh=0.45):
    # dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
    # thresh:0.3,0.5....
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bbox
    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)
        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= thresh)[0]
        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    return keep


# 定义sigmod函数
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# 检测模型前向运算
def Load_YOLO_model(net, test_img, output_names):
    input_img = padimg2(test_img)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 0, 1)
    net.blobs["input"].data[...] = input_img  # images, data
    out = net.forward()
    outputs = []
    for i in range(NUM_STRIDES):
        outputs.append(out[output_names[i]].transpose(0, 3, 2, 1)[0])  # no reshape
    return outputs


# 处理前向输出feature_map
def feature_map_handle(length, pred, box_list):
    global INPUT_WIDTH_PADDED, INPUT_HEIGHT_PADDED
    ih, iw = INPUT_HEIGHT_PADDED, INPUT_WIDTH_PADDED
    coefx = float(iw) / INPUT_WIDTH
    coefy = float(ih) / INPUT_HEIGHT
    featw, feath = pred.shape[0], pred.shape[1]
    confidence = 0.25

    feathnew = featw * float(INPUT_HEIGHT) / INPUT_WIDTH
    featwnew = feath * float(INPUT_HEIGHT) / INPUT_WIDTH

    stride = [32, 16, 8]

    feature_map_size1 = int(INPUT_WIDTH / stride[0])  # 15 #
    feature_map_size2 = int(INPUT_WIDTH / stride[1])  # 30
    feature_map_size3 = int(INPUT_WIDTH / stride[2])  # 60

    fh1 = int(INPUT_HEIGHT / stride[0])  # 9
    fh2 = int(INPUT_HEIGHT / stride[1])  # 18
    fh3 = int(INPUT_HEIGHT / stride[2])  # 36

    for i in range(feath):  # length
        for j in range(featw):  # length
            anchors_box = pred[j][i]
            # 计算实际置信度,阀值处理,anchors_box[7]
            score = sigmoid(anchors_box[4])
            # print('score', score, anchors_box[4])
            if score > confidence:
                # tolist()数组转list
                cls_list = anchors_box[5 : CLASS_NUM + 5].tolist()
                label = cls_list.index(max(cls_list))
                # obj_score = score

                clsscore = sigmoid(cls_list[label])
                obj_score = score * clsscore

                if obj_score < confidence:
                    # print('what the fuck')
                    continue

                x = ((sigmoid(anchors_box[0]) * 2 - 0.5 + j) / float(featw)) * iw
                y = ((sigmoid(anchors_box[1]) * 2 - 0.5 + i) / float(feathnew)) * ih

                if length == feature_map_size1:  # stride=32
                    bw, bh = ANCHORS[4], ANCHORS[5]
                    w = math.pow(sigmoid(anchors_box[2]) * 2, 2) * bw  # *iw/imgWidth
                    h = math.pow(sigmoid(anchors_box[3]) * 2, 2) * bh  # *ih/imgHeight
                elif length == feature_map_size2:  # stride=16
                    bw, bh = ANCHORS[2], ANCHORS[3]
                    w = math.pow(sigmoid(anchors_box[2]) * 2, 2) * bw  # *iw/imgWidth
                    h = math.pow(sigmoid(anchors_box[3]) * 2, 2) * bh  # *ih/imgHeight
                elif length == feature_map_size3:  # stride=8
                    bw, bh = ANCHORS[0], ANCHORS[1]
                    w = math.pow(sigmoid(anchors_box[2]) * 2, 2) * bw  # *iw/imgWidth
                    h = math.pow(sigmoid(anchors_box[3]) * 2, 2) * bh  # *ih/imgHeight

                w *= coefx
                h *= coefy

                x1 = int(x - w * 0.5)
                x2 = int(x + w * 0.5)
                y1 = int(y - h * 0.5)
                y2 = int(y + h * 0.5)

                if LABEL_OFFSET:
                    nmsoffset = label * MAX_WH
                    x1 += nmsoffset
                    y1 += nmsoffset
                    x2 += nmsoffset
                    y2 += nmsoffset

                box_list.append([x1, y1, x2, y2, round(obj_score, 4), label])


# 3个feature_map的预选框的合并及NMS处理
def dect_box_handle(pred):
    box_list = []
    output_box = []
    for i in range(NUM_STRIDES):
        length = len(pred[i])
        feature_map_handle(length, pred[i], box_list)
    # print box_list
    if box_list:
        retain_box_index = nms(np.array(box_list))
        for i in retain_box_index:
            output_box.append(box_list[i])
    return output_box



if __name__ == "__main__":
    # 类别数目
    CLASS_NUM = 3
    # 加载label文件
    LABEL_NAMES = ["person", "two-wheeler", "car"]

    stride = [8, 16, 32]

    ANCHORS = [11.2109375, 12.7578125, 28.09375, 45.59375, 143.0, 120.5]
    # BIAS_W = [10, 27, 33, 80, 62, 87, 177, 218, 382]
    # BIAS_H = [13, 16, 42, 56, 109, 119, 151, 281, 261]

    output_names = ["output.1", "output.2", "output.3"]
    netname = "runs/build/yolox_cspm_coc_f/yolox_cspm.prototxt"
    modelname = "runs/build/yolox_cspm_coc_f/yolox_cspm.caffemodel"

    net = caffe.Net(netname, modelname, caffe.TEST)
    print(netname, modelname)

    #################################
    source_path = "../datasets/fuh-testpic"
    output_path = "runs/test"

    shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in os.listdir(source_path):
        img_path = os.path.join(source_path, file)
        test_img = cv2.imread(img_path)
        if test_img is None:
            print('warning: ignore invalid image', test_img)
            continue
        print(img_path)

        pred = Load_YOLO_model(net, test_img, output_names)
        output_box = dect_box_handle(pred)
        for i in output_box:
            if LABEL_OFFSET:
                label = i[5]
                nmsoffset = label * MAX_WH
                i[0] -= nmsoffset
                i[1] -= nmsoffset
                i[2] -= nmsoffset
                i[3] -= nmsoffset
            if LABEL_NAMES[i[5]] == "person":
                cv2.rectangle(test_img, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 3)
            else:
                cv2.rectangle(test_img, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), 3)
            cv2.circle(test_img, (int(i[0] + 0.5 * (i[2] - i[0])), int(i[1] + 0.5 * (i[3] - i[1]))), 2, (0, 255, 0), 3)
            cv2.putText(test_img, "Score:" + str(i[4]), (i[0], i[1] - 5), 0, 0.7, (255, 0, 255), 2)
            cv2.putText(test_img, "Label:" + str(LABEL_NAMES[i[5]]), (i[0], i[1] - 20), 0, 0.7, (255, 255, 0), 2)
        cv2.imwrite(output_path + "/" + file, test_img)
