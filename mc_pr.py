# -*- coding: utf-8 -*-

########################################################################
# #
# #    Copyright: Copyright (c) 2020
# #    Created on 2020-06-17
# #    Author: 穆翀
# #    Version 1.0
# #    Title: mc-evaluation_of_recall_and_precision_in_object_detection.py
# #
########################################################################

'''
 -*- mc -*-
说明：
    这个代码的目的是：
        在目标检测任务中如何计算评价指标 Recall、Precision以及画出PR曲线。
        最近在评价结果的时候发现目标检测的评价结果还是比较难以考虑的

        和单纯的计算Recall和Precision相比，多类的目标检测的Recall和Precision计算有如下几个难度：
        1，二类问题的Recall和Precision容易解决，那多类问题该怎么办呢
        2，分类任务的评价指标还是容易计算的，那检测任务呢，如何计算其TP以及FP?
        3，网上有很多解决方案都是画一个类别的PR曲线，如何将多类别的曲线用一条PR曲线画出来呢？

        True positives: 简称为TP，即正样本被正确识别为正样本，飞机的图片被正确的识别成了飞机。
        False Positives: 简称为FP，即负样本被错误识别为正样本，大雁的图片被错误地识别成了飞机。
        True negatives: 简称为TN，即负样本被正确识别为负样本，大雁的图片没有被识别出来，系统正确地认为它们是大雁。
        False negatives: 简称为FN，即正样本被错误识别为负样本，飞机的图片没有被识别出来，系统错误地认为它们是大雁。

'''

# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
#import cPickle
import pickle
import numpy as np
import cv2 as cv
#import sys
#import importlib
#importlib.reload(sys)

# data name
# data name
OBJLABEL = ["bengbian", "duanshan", "loujiang", "zangpian"]
GTNU_num = {"bengbian":0, "duanshan":0, "loujiang":0, "zangpian":0}
YOLO_num = {"bengbian":0, "duanshan":0, "loujiang":0, "zangpian":0}
POSITIVE_num = {"bengbian":0, "duanshan":0, "loujiang":0, "zangpian":0}


# 设置交并比阈值
IOUthreshold = 0.8

def parse_xml(filename):
    global GTNU_num,YOLO_num,POSITIVE_num
    """ 解析 xml 文件 """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

        # 统计ground truth中，每个标注对象类别的数量
        if obj_struct['name'] in OBJLABEL:
            GTNU_num[obj_struct['name']] +=1

    print(objects)

    return objects

def vis_label(imgname, objects, mc_save_vis_label):
    """ 将解析后的 xml 文件，在原图上进行可视化处理 """
    # imgname 原图
    # 解析后的 xml 文件
    img = cv.imread(imgname)

    if len(objects) == 0:
        print("请检查 " + str(imgname) + " 的标签！！！")
        return

    for i in range(len(objects)):
        try:
            objectname = objects[i]['name']

            x1 = objects[i]['bbox'][0]
            y1 = objects[i]['bbox'][1]
            x2 = objects[i]['bbox'][2]
            y2 = objects[i]['bbox'][3]

            minX = x1
            minY = y1
            maxX = x2
            maxY = y2

            # print(objectname, minX, minY, maxX, maxY)

            # BB 绘制
            color = (255, 0, 0)
            cv.rectangle(img, (minX, minY), (maxX, maxY), color, 3)

            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img, objectname, (minX, minY - 7), font, 1, (0, 0, 255), 2)

        except:
            print('错误')

    # 显示图片
    # cv.imshow("mc label", img)
    # cv.waitKey()

    # 保存图片
    # path = os.path.join(save_path, image_name)
    cv.imwrite(mc_save_vis_label, img)

def parse_yolo_result(txtname):
    """ 将事先保存的YOLO预测的结果，进行解析 """
    # txt保存YOLO预测的结果格式 图片名称为txt文件名 单行内容为 类别名 置信度 minX minY maxX maxY

    global GTNU_num,YOLO_num,POSITIVE_num

    with open(txtname, 'r') as file:
        # 逐行处理
        lines = file.readlines()
    yolo_obj = [x.strip() for x in lines]
    # print(yolo_obj)

    res_objects = []

    for i in range(len(yolo_obj)):
        information = yolo_obj[i].split(' ')
        # _obj_name = information[0].split("'")
        obj_struct = {}
        # obj_struct['name'] = str(_obj_name[1])
        obj_struct['name'] = str(information[0])
        obj_struct['confidence'] = float(information[1])
        # print(obj_struct['confidence'].__class__)
        # print(information[2].__class__)
        obj_struct['bbox'] = [int(float(information[2])),
                              int(float(information[3])),
                              int(float(information[4])),
                              int(float(information[5]))]
        res_objects.append(obj_struct)

        # 统计YOLO检测结果中，每个检测出的对象类别的数量
        YOLO_num[obj_struct['name']] +=1

    print(res_objects)

    return res_objects

def vis_label_and_yolo(imgname, objects, objects_yolo, mc_save_vis_label_and_yolo):
    """ 将解析后的 xml 文件，在原图上进行可视化处理 """
    # imgname 原图
    # 解析后的 xml 文件
    img = cv.imread(imgname)

    if len(objects) == 0 or len(objects_yolo) == 0:
        print("请检查 " + str(imgname) + " 的标签和YOLO检测的结果！！！")
        return

    # Ground Truth
    for i in range(len(objects)):
        try:
            objectname = objects[i]['name']

            x1 = objects[i]['bbox'][0]
            y1 = objects[i]['bbox'][1]
            x2 = objects[i]['bbox'][2]
            y2 = objects[i]['bbox'][3]

            minX = x1
            minY = y1
            maxX = x2
            maxY = y2

            # print(objectname, minX, minY, maxX, maxY)

            # BB 绘制
            color = (255, 0, 0)
            cv.rectangle(img, (minX, minY), (maxX, maxY), color, 3)

            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img, objectname, (minX, minY - 7), font, 1, (0, 0, 255), 2)

        except:
            print('Ground Truth 错误')

    # YOLO 检测结果
    for j in range(len(objects_yolo)):
        try:
            objectname_ = objects_yolo[j]['name']

            x1_ = objects_yolo[j]['bbox'][0]
            y1_ = objects_yolo[j]['bbox'][1]
            x2_ = objects_yolo[j]['bbox'][2]
            y2_ = objects_yolo[j]['bbox'][3]

            minX_ = x1_
            minY_ = y1_
            maxX_ = x2_
            maxY_ = y2_

            # BB 绘制
            color_ = (255, 0, 255)
            cv.rectangle(img, (minX_, minY_), (maxX_, maxY_), color_, 3)

            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img, objectname_, (minX_ - 7, minY_ - 7), font, 1, (0, 255, 255), 2)

        except:
            print('YOLO 检测结果 错误')

    # 显示图片
    # cv.imshow("mc label_and_yolo", img)
    # cv.waitKey()

    # 保存图片
    # path = os.path.join(save_path, image_name)
    cv.imwrite(mc_save_vis_label_and_yolo, img)

def judge_rectangles_intersect(p1, p2):
    """ 判断两个矩形是否相交
    如何快速判断两个矩形相交：
                如果两个矩形相交，那么矩形A B的中心点和矩形的边长是有一定关系的。
                两个矩形中心点间的距离肯定小于AB边长和的一半。

        设A[x01,y01,x02,y02]  B[x11,y11,x12,y12].
        矩形A和矩形B物理中心点X方向的距离为Lx：abs( (x01+x02)/2 – (x11+x12) /2)
        矩形A和矩形B物理中心点Y方向的距离为Ly：abs( (y01+y02)/2 – (y11+y12) /2)
        矩形A和矩形B X方向的边长为 Sax：abs(x01-x02)  Sbx: abs(x11-x12)
        矩形A和矩形B Y方向的边长为 Say：abs(y01-y02)  Sby: abs(y11-y12)
        如果AB相交，则满足下列关系：
        Lx <= (Sax + Sbx)/2 && Ly <=(Say+ Sby)/2
    """

    if len(p1) == 0 or len(p2) == 0:
        print("矩形判断模块报错！")
        return 0

    p1minX = p1[0]
    p1minY = p1[1]
    p1maxX = p1[2]
    p1maxY = p1[3]

    p2minX = p2[0]
    p2minY = p2[1]
    p2maxX = p2[2]
    p2maxY = p2[3]

    x_distance = abs(p1minX + p1maxX - p2minX - p2maxX)
    x_length = abs(p1minX - p1maxX) + abs(p2minX - p2maxX)
    y_distance = abs(p1minY + p1maxY - p2minY - p2maxY)
    y_length = abs(p1minY - p1maxY) + abs(p2minY - p2maxY)

    if x_distance < x_length and y_distance < y_length:
        return 1
    else:
        return 0

def compute_iou(rectangle1, rectangle2):
    """ 计算两个相交矩形的交并比 """
    global IOUthreshold

    # 计算IOU compute overlaps
    # 交集 intersection
    ixmin = np.maximum(rectangle1[0], rectangle2[0])
    iymin = np.maximum(rectangle1[1], rectangle2[1])
    ixmax = np.minimum(rectangle1[2], rectangle2[2])
    iymax = np.minimum(rectangle1[3], rectangle2[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # 并集 union
    uni = ((abs(rectangle1[0] - rectangle1[2]) + 1.) * (abs(rectangle1[1] - rectangle1[3]) + 1.) +
           (abs(rectangle2[0] - rectangle2[2]) + 1.) * (abs(rectangle2[1] - rectangle2[3]) + 1.)
           - inters)

    overlaps = inters / uni
    print(overlaps)

    if overlaps > IOUthreshold:
        return 1
    else:
        return 0

def compared_gt_and_res(gt_objects, res_objects):
    """ 将ground truth解析后结果，与YOLO预测解析后的结果，进行比对判断 """

    global GTNU_num,YOLO_num,POSITIVE_num

    # ground truth
    for i in range(len(gt_objects)):
        try:
            objectname = gt_objects[i]['name']

            x1 = gt_objects[i]['bbox'][0]
            y1 = gt_objects[i]['bbox'][1]
            x2 = gt_objects[i]['bbox'][2]
            y2 = gt_objects[i]['bbox'][3]

            minX = x1
            minY = y1
            maxX = x2
            maxY = y2

            gt_list = [minX, minY, maxX, maxY]

            # YOLO预测解析后的结果
            for j in range(len(res_objects)):
                yolo_objectname = res_objects[j]['name']
                yolo_x1 = res_objects[j]['bbox'][0]
                yolo_y1 = res_objects[j]['bbox'][1]
                yolo_x2 = res_objects[j]['bbox'][2]
                yolo_y2 = res_objects[j]['bbox'][3]

                yolo_list = [yolo_x1, yolo_y1, yolo_x2, yolo_y2]

                # 判断两个矩形是否相交
                if judge_rectangles_intersect(gt_list, yolo_list):
                    print(gt_list, yolo_list)
                    if compute_iou(gt_list, yolo_list) == 1:
                        # 如果交并比满足阈值条件，则对标签一致性进行比较
                        if objectname == yolo_objectname:
                            # 如果标签一致，则统计正例个数
                            # 统计YOLO检测结果中，每个被正确检测出的对象类别的数量
                            POSITIVE_num[objectname] +=1

                            # print("正例统计")
                    # return

        except:
            print('错误')

def vis_error(mc_img_path, gt_objects, res_objects, mc_save_error):
    """ 可视化 误检目标 """
    # imgname 原图
    img = cv.imread(mc_img_path)

    if len(gt_objects) == 0 or len(res_objects) == 0:
        print("请检查 " + str(mc_img_path) + " 的标签和YOLO检测的结果！！！")
        return

    # ground truth
    for i in range(len(gt_objects)):
        try:
            objectname = gt_objects[i]['name']

            x1 = gt_objects[i]['bbox'][0]
            y1 = gt_objects[i]['bbox'][1]
            x2 = gt_objects[i]['bbox'][2]
            y2 = gt_objects[i]['bbox'][3]

            minX = x1
            minY = y1
            maxX = x2
            maxY = y2

            gt_list = [minX, minY, maxX, maxY]

            # YOLO预测解析后的结果
            for j in range(len(res_objects)):
                yolo_objectname = res_objects[j]['name']
                yolo_x1 = res_objects[j]['bbox'][0]
                yolo_y1 = res_objects[j]['bbox'][1]
                yolo_x2 = res_objects[j]['bbox'][2]
                yolo_y2 = res_objects[j]['bbox'][3]

                # 首次绘图 绘制网络输出结果
                if i == 0:
                    # BB 绘制
                    color_ = (0, 0, 255)
                    cv.rectangle(img, (yolo_x1, yolo_y1), (yolo_x2, yolo_y2), color_, 2)

                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(img, yolo_objectname, (yolo_x1 - 7, yolo_y1 - 7), font, 1, (0, 0, 255), 2)

                yolo_list = [yolo_x1, yolo_y1, yolo_x2, yolo_y2]

                # 判断两个矩形是否相交
                if judge_rectangles_intersect(gt_list, yolo_list):
                    print(gt_list, yolo_list)
                    if compute_iou(gt_list, yolo_list) == 1:
                        # 如果交并比满足阈值条件，则对标签一致性进行比较
                        if objectname == yolo_objectname:
                            # 如果标签一致，则统计正例个数
                            # 统计YOLO检测结果中，每个被正确检测出的对象类别的数量
                            if objectname == OBJLABEL[0]:
                                # BB 绘制被正确检测出的对象
                                color = (255, 0, 0)
                                cv.rectangle(img, (yolo_x1, yolo_y1), (yolo_x2, yolo_y2), color, 3)
                            elif objectname == OBJLABEL[1]:
                                # BB 绘制被正确检测出的对象
                                color = (255, 0, 0)
                                cv.rectangle(img, (yolo_x1, yolo_y1), (yolo_x2, yolo_y2), color, 3)
                            elif objectname == OBJLABEL[2]:
                                # BB 绘制被正确检测出的对象
                                color = (255, 0, 0)
                                cv.rectangle(img, (yolo_x1, yolo_y1), (yolo_x2, yolo_y2), color, 3)

                            # print("正例统计")
                    # return

        except:
            print('错误')

    # 显示图片
    # cv.imshow("mc error", img)
    # cv.waitKey()

    # 保存图片
    # path = os.path.join(save_path, image_name)
    cv.imwrite(mc_save_error, img)

def eval_recall_and_precision():
    """ 准确率和召回率 统计计算 """
    gt_all_num = 0
    yolo_all_num = 0
    precision_all_num =0
    for key,val in GTNU_num.items():
        gt_all_num +=val
    for key,val in YOLO_num.items():
        yolo_all_num +=val
    for key,val in POSITIVE_num.items():
        precision_all_num +=val

    recall_all = yolo_all_num / gt_all_num
    print("recall_all:---" + str(recall_all))
    #print('recall')
  #  print(recall_all)

    precision_all = precision_all_num / yolo_all_num
    print("precision_all :---" + str(precision_all))
 
    recall_obj = {"bengbian":0, "duanshan":0, "loujiang":0, "zangpian":0}
    for key,val in recall_obj.items():
        if GTNU_num[key]>0:
            recall_obj[key] = YOLO_num[key]/GTNU_num[key]
            print('recall:'+str(key)+'---'+str(recall_obj[key]))
    print('--------precision------')
    precision_obj ={"bengbian":0, "duanshan":0, "loujiang":0, "zangpian":0}
    for key,val in precision_obj.items():
        if GTNU_num[key]>0:
            precision_obj[key] = POSITIVE_num[key]/YOLO_num[key]
            print('precision:'+str(key)+'---'+str(precision_obj[key]))

   

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# 指标评价函数
def mc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)

    detpath: 产生的txt文件，里面是一张图片的各个检测框结果。
    annopath: xml 文件与对应的图像相呼应。
    imagesetfile: 一个txt文件，里面是每个图片的地址，每行一个地址。
    classname: 种类的名字，即类别。
    cachedir: 缓存标注的目录。
    [ovthresh]: IOU阈值，默认为0.5，即mAP50。
    [use_07_metric]: 是否使用2007的计算AP的方法，默认为Fasle

    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # 首先加载Ground Truth标注信息
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print
                'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print
        'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def main():

    # 文件路径设置
    # Ground Truth 路径
    #filename = "/notebooks/home/shixiaomeng/darknet-master/my_data/pinggu_v4/test_xml"
    filename = "F:\\fastercnn\\default_resnet\\xml-txt"
    # 测试图片路径
    #imgname = "/notebooks/home/shixiaomeng/darknet-master/my_data/pinggu_v4/test_image"
    imgname = "F:\\fastercnn\\data\\demoall"
    # YOLO 检测结果路径
    #txtname = "/notebooks/home/shixiaomeng/yolov4opencv/darknet-master/yolov4-23002-output-txt-0.9-416-0623"
    txtname = "F:\\fastercnn\\default_resnet\\test-txt\\"
   # txtname = "/home/wangshuang/project/Yolo_mc_test/v4_tiny/test-txt"

    # 可视化Ground Truth 保存路径
    save_vis_label = "F:\\fastercnn\\default_resnet\\v4tiny-vis_label-0.5-test"
    if not os.path.exists(save_vis_label):
        os.makedirs(save_vis_label)

    # 可视化 Ground Truth 和 YOLO 检测结果 保存路径
    save_vis_label_and_yolo = "F:\\fastercnn\\default_resnet\\v4tiny-vis_label_and_yolo-0.5-test"
    if not os.path.exists(save_vis_label_and_yolo):
        os.makedirs(save_vis_label_and_yolo)

    # 可视化 误检结果 保存路径
    save_vis_error = "F:\\fastercnn\\default_resnet\\v4tiny-vis_error-0.5-test"
    if not os.path.exists(save_vis_error):
        os.makedirs(save_vis_error)

    # 读入文件
    # &&&&&&&&&&&&& 注意 要求 Ground Truth、测试图片、YOLO 检测结果 在三个文件夹中，各自文件一一对应 &&&&&&&&&&&&&&
    xml_list = os.listdir(filename)

    # 读取并处理每一套 （三个）Ground Truth、测试图片、YOLO 检测结果 文件
    for i in range(len(xml_list)):
        print("Processing the " + str(i + 1) + " th image and label, " + str(len(xml_list)) + " total.")

        # 分离名称
        name, ext = os.path.splitext(xml_list[i])

        mc_gt = name + ".xml"
        mc_img = name + ".jpg"
        mc_txt = name + ".txt"

        # Ground Truth全路径
        mc_gt_path = os.path.join(filename, mc_gt)
        # 测试图片全路径
        mc_img_path = os.path.join(imgname, mc_img)
        # YOLO 检测结果全路径
        mc_txt_path = os.path.join(txtname, mc_txt)

        # 可视化Ground Truth 保存路径
        mc_save_vis_label = os.path.join(save_vis_label, mc_img)
        # 可视化 Ground Truth 和 YOLO 检测结果 保存路径
        mc_save_vis_label_and_yolo = os.path.join(save_vis_label_and_yolo, mc_img)
        # 可视化误检目标 保存路径
        mc_save_error = os.path.join(save_vis_error, mc_img)

        # 解析 Ground Truth XML文件
        temp1 = parse_xml(mc_gt_path)

        # 可视化 Ground Truth
        vis_label(mc_img_path, temp1, mc_save_vis_label)

        # 解析 YOLO 检测结果 TXT文件
        temp2 = parse_yolo_result(mc_txt_path)

        # 可视化 Ground Truth 和 YOLO 检测结果
        vis_label_and_yolo(mc_img_path, temp1, temp2, mc_save_vis_label_and_yolo)

        # 将ground truth解析后结果，与YOLO预测解析后的结果，进行比对判断
        compared_gt_and_res(temp1, temp2)

        # 可视化检测错误的目标
        # 将检测出的结果，使用颜色1进行可视化 然后将比对结果，用颜色2可视化到同一张图片中，则图片仍然为颜色1的目标，则为误检目标。
        vis_error(mc_img_path, temp1, temp2, mc_save_error)

    for key,val in GTNU_num.items():
        print("In dataset Ground Truth classes are:" + str(key) + ",result numbers are:--" + str(val))
    for key,val in YOLO_num.items():
        print("In dataset YOLO detection classes are:" + str(key) + ",result numbers are:--" + str(val))
    for key,val in POSITIVE_num.items():
        print("In dataset YOLO correctly  detection classes are:" + str(key) + ",result numbers are:--" + str(val))

    eval_recall_and_precision()

if __name__ == '__main__':
    main()