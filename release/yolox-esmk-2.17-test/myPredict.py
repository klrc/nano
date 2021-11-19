# -*- coding: UTF-8 -*-
from __future__  import division
import math
import sys

import argparse
import os
import json
import time
import sys
# sys.path.append('.')
import logging
import cv2
import torch
import torch.nn.parallel

from fake_loader import attempt_load

import cv2
import os


param_file_path = 'yolox-esmk.pt'

model = attempt_load(weights=param_file_path, device='cpu') 
model.eval()


print('anchors', model.head.anchors)
print('begin forward...')


imname = 'C:/Users/frank.tu/Documents/testpic_fh/testpic/personcar/car_persion_1.jpeg'


im = cv2.imread(imname)
# cv2.imshow("im", im)
# cv2.waitKey()

imgWidth = 416
imgHeight = 224

input_img = cv2.resize(im, (imgWidth, imgHeight))
dispimg = input_img.copy()

input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

input_img = input_img/255.0

input_img = input_img.transpose(2,0,1)


input_img = torch.tensor([input_img])
input_img = input_img.float()

out = model.inference(input_img)

print(out[0].shape)
print(out[1].shape)
print(out[2].shape)

numAnchor = 1
CLASS_NUM = 3
confidence = 0.1
box_list = []
numstride = 3


for sid in range(numstride):
	shapesize = out[sid].shape
	featw, feath = shapesize[3],shapesize[2]
	shape = out[sid][0][0]

	for i in range(feath):#length
	        for j in range(featw): #length
	            # anchors_boxs_shape = shape[j][i] #.reshape((3, CLASS_NUM + 5))
	            anchors_boxs_shape = shape[i][j].detach().numpy() #.reshape((numAnchor, CLASS_NUM + 5))
	            #将每个预测框向量包含信息迭代出来
	            # print('anchors_boxs_shape',anchors_boxs_shape)
	            for k in range(numAnchor):
	                anchors_box = anchors_boxs_shape #[k]
	                #计算实际置信度,阀值处理,anchors_box[7]
	                # print('anchors_box', anchors_box)
	                # input()
	                score = (anchors_box[4])
	                # print('score', score, anchors_box[4])
	                if score > confidence:
	                    #tolist()数组转list
	                    cls_list = anchors_box[5:CLASS_NUM + 5].tolist()
	                    label = cls_list.index(max(cls_list))
	                    # obj_score = score
	                    
	                    clsscore = (cls_list[label])
	                    obj_score = score*clsscore

	                    if(obj_score < confidence):
	                        # print('what the fuck')
	                        continue

	                    # x = ((sigmod(anchors_box[0]) + i)/float(featw))*iw #length
	                    # y = ((sigmod(anchors_box[1]) + j)/float(feath))*ih
	                    # x = ((sigmod(anchors_box[0]) + i)/float(feath))*iw #length
	                    # y = ((sigmod(anchors_box[1]) + j)/float(featwnew))*ih
	    
	                    # x = sigmod(anchors_box[0])*2 #- 0.5
	                    # y = sigmod(anchors_box[1])*2 #- 0.5

	                    x = anchors_box[0]*imgWidth #((sigmod(anchors_box[0])*2 - 0.5 + j)/float(featw))*iw
	                    y = anchors_box[1]*imgHeight #((sigmod(anchors_box[1])*2 - 0.5 + i)/float(feathnew))*ih
	                    w = anchors_box[2]*imgWidth
	                    h = anchors_box[3]*imgHeight
	                    print(x, y, w, h)
	                    x1 = int(x - w * 0.5)
	                    x2 = int(x + w * 0.5)
	                    y1 = int(y - h * 0.5)
	                    y2 = int(y + h * 0.5)

	                    box_list.append([x1,y1,x2,y2,round(obj_score,4),label])

print('box_list', box_list)
test_img = dispimg
for i in box_list:
    # if not (LABEL_NAMES[i[5]] == 'person' or LABEL_NAMES[i[5]]=='car'):
    #   continue
    # print(LABEL_NAMES[i[5]], i[4])
    # cv2.rectangle(test_img, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), 3)
    # if LABEL_NAMES[i[5]] == 'person' or LABEL_NAMES[i[5]]=='car' or LABEL_NAMES[i[5]]=='bus' or LABEL_NAMES[i[5]]=='truck':
    #     cv2.rectangle(test_img, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 3)
    # else:
    cv2.rectangle(test_img, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), 3)
    cv2.circle(test_img, (int(i[0]+0.5*(i[2]-i[0])), int(i[1]+0.5*(i[3]-i[1]))), 2, (0,255,0), 3)           
    cv2.putText(test_img, "Score:"+str(i[4]), (i[0], i[1]-5), 0, 0.7, (255, 0, 255), 2) 
    # cv2.putText(test_img, "Label:"+str(LABEL_NAMES[i[5]]), (i[0], i[1]-20), 0, 0.7, (255, 255, 0), 2)

cv2.imshow('img', test_img)
cv2.waitKey()
print('done')