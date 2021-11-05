# -*- coding: UTF-8 -*-
from __future__ import division
import math
import caffe
import numpy as np
import cv2
import os
import shutil


def pre_padding(img, std_coef):
    # 填充图像边缘，使长宽比与输出长宽比保持一致
    raw_size = [x for x in img.shape[:2]]
    expand_size = [x for x in img.shape[:2]]
    coef = raw_size[1] * 1.0 / raw_size[0]
    if coef > std_coef:
        expand_size[0] = int(expand_size[0] / std_coef)
    else:
        expand_size[1] = int(expand_size[1] * std_coef)
    result = np.zeros((expand_size[0], expand_size[1], 3), np.uint8)
    result[: raw_size[0], : raw_size[1]] = img
    return result


def pre_resize(img, size):
    return cv2.resize(img, size)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


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


# 处理前向输出feature_map
def detect_head(pred, anchors, input_size, num_classes, confidence = 0.25):

    result = []
    ih, iw = input_size
    featw, feath = pred.shape[0], pred.shape[1]

    for i in range(feath):  # length
        for j in range(featw):  # length
            anchors_box = pred[j][i]
            # 计算实际置信度,阀值处理,anchors_box[7]
            score = sigmoid(anchors_box[4])
            # print('score', score, anchors_box[4])
            if score > confidence:
                # tolist()数组转list
                cls_list = anchors_box[5 : num_classes + 5].tolist()
                label = cls_list.index(max(cls_list))
                clsscore = sigmoid(cls_list[label])
                obj_score = score * clsscore

                x = ((sigmoid(anchors_box[0]) * 2 - 0.5 + j) / float(featw)) * iw
                y = ((sigmoid(anchors_box[1]) * 2 - 0.5 + i) / float(feath)) * ih
                w = math.pow(sigmoid(anchors_box[2]) * 2, 2) * anchors[0]
                h = math.pow(sigmoid(anchors_box[3]) * 2, 2) * anchors[1]

                x1 = int(x - w * 0.5)
                x2 = int(x + w * 0.5)
                y1 = int(y - h * 0.5)
                y2 = int(y + h * 0.5)

                nmsoffset = label * 14096  # for nms offset
                x1 += nmsoffset
                y1 += nmsoffset
                x2 += nmsoffset
                y2 += nmsoffset

                result.append([x1, y1, x2, y2, round(obj_score, 4), label])
    return result


# 检测模型前向运算
def inference(net, test_img, output_names, input_shape=(224, 416), num_strides=3, num_classes=3):
    test_img = pre_padding(test_img, input_shape[1] * 1.0 / input_shape[0])
    input_img = pre_resize(test_img, input_shape)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 1, 0)
    net.blobs["input"].data[...] = input_img  # images, data
    out = net.forward()
    preds = []
    for i in range(num_strides):
        preds.append(out[output_names[i]].transpose(0, 3, 2, 1)[0])  # no reshape
    box_list = []
    output_box = []
    for i, pred in enumerate(preds):
        result = detect_head(pred, anchors[2*i: 2*i+2], input_shape, num_classes=num_classes)
        box_list.extend(result)
    if box_list:
        retain_box_index = nms(np.array(box_list))
        for i in retain_box_index:
            output_box.append(box_list[i])
    return output_box


if __name__ == "__main__":

    proto_path = "runs/build/yolox_cspm_coc_f/yolox_cspm.prototxt"
    weight_path = "runs/build/yolox_cspm_coc_f/yolox_cspm.caffemodel"
    output_names = ["output.1", "output.2", "output.3"]
    label_names = ["person", "two-wheeler", "car"]

    input_shape = (224, 416)
    strides = [8, 16, 32]
    anchors = [11.2109375, 12.7578125, 28.09375, 45.59375, 143.0, 120.5]

    net = caffe.Net(proto_path, weight_path, caffe.TEST)
    print(proto_path, weight_path)

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
            print("warning: ignore invalid image", test_img)
            continue
        print(img_path)

        output_box = inference(net, test_img, output_names, input_shape, len(strides))
        for i in output_box:
            label = i[5]
            nmsoffset = label * 14096  # for nms offset
            i[0] -= nmsoffset
            i[1] -= nmsoffset
            i[2] -= nmsoffset
            i[3] -= nmsoffset
            if label_names[i[5]] == "person":
                cv2.rectangle(test_img, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 3)
            else:
                cv2.rectangle(test_img, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), 3)
            cv2.circle(test_img, (int(i[0] + 0.5 * (i[2] - i[0])), int(i[1] + 0.5 * (i[3] - i[1]))), 2, (0, 255, 0), 3)
            cv2.putText(test_img, "Score:" + str(i[4]), (i[0], i[1] - 5), 0, 0.7, (255, 0, 255), 2)
            cv2.putText(test_img, "Label:" + str(label_names[i[5]]), (i[0], i[1] - 20), 0, 0.7, (255, 255, 0), 2)
        cv2.imwrite(output_path + "/" + file, test_img)
