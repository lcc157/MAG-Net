# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

####################################################################
from PyQt5 import QtWidgets, QtGui
import sys
from first import Ui_Dialog   # 导入生成first.py里生成的类
from PyQt5.QtWidgets import QFileDialog

CLASSES = ('__background__',
           'finger')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

#define Global data
class MyGlobal:
    def __init__(self):
        openimageName = ''
        Running = 1
GL = MyGlobal()

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im_file = os.path.join(image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args


class mywindow(QtWidgets.QWidget,Ui_Dialog):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        #定义槽函数

    def openimage(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        global penimageName
        GL.openimageName, imgType = QFileDialog.getOpenFileNames(self,
                                                       "打开图片",
                                                       "",
                                                       " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")

        print(GL.openimageName)

    def PAUSE(self):
        GL.Runing = 0



    def Start(self):

        GL.Runing = 1
        args = parse_args()

        # model path
        demonet = args.demo_net
        dataset = args.dataset
        # tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])
        tfmodel = 'default/voc_2007_trainval/default/vgg16_faster_rcnn_iter_30000.ckpt'
        if not os.path.isfile(tfmodel + '.meta'):
            print(tfmodel)
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                           'our server and place them properly?').format(tfmodel + '.meta'))

        # set config

        num_gpus = 0
        num_cpus = 1  # num_cpus =表示同时使用cpu个数,且num_cpus>=1
        config = tf.ConfigProto(
            device_count={'GPU': num_gpus, 'CPU': num_cpus},
            allow_soft_placement=True,  # 自动选中GPU
            log_device_placement=False  # 打印设备分配日志
        )

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True

        # init session
        sess = tf.Session(config=tfconfig)
        # load network
        if demonet == 'vgg16':
            net = vgg16(batch_size=1)
        # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
        else:
            raise NotImplementedError
        net.create_architecture(sess, "TEST", 2,
                                tag='default', anchor_scales=[8, 16, 32])
        saver = tf.train.Saver()
        saver.restore(sess, tfmodel)

        print('Loaded network {:s}'.format(tfmodel))

        # im_names = ['000456.jpg', '000457.jpg', '000542.jpg', '001150.jpg',
        #             '001763.jpg', '004545.jpg']
        #im_names = ['1.jpg', '2.jpg', '3.jpg', '4.jpg',
        #            '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg']

        print(GL.openimageName)
        im_names = GL.openimageName
        i=1
        while GL.Runing == 1:
            for im_name in im_names:
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
               # print('Demo for data/demo/{}'.format(im_name))
                print(format(im_name))
                demo(sess, net, im_name)
                plt.savefig("test_save/finish" + str(i)+".jpg")
                print(i)

                #first label
                imgName = os.path.join(format(im_name))
                print(imgName)
                # 利用qlabel显示图片
                png = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(png)
                QtWidgets.QApplication.processEvents()

                # second label
                imgName2 = "test_save/finish" + str(i)+".jpg"
#                   print(imgName)
                # 利用qlabel显示图片
                png2 = QtGui.QPixmap(imgName2).scaled(self.label_2.width(), self.label_2.height())
                self.label_2.setPixmap(png2)
                QtWidgets.QApplication.processEvents()
                i = i + 1
                if i > len(GL.openimageName):
                    i = 1

#        plt.show()



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())
