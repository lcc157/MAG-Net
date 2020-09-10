

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
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
from lib.utils.test import im_detect,test_net
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.nets.resnet_v1 import resnetv1
from lib.nets.resnet_v1_att import resnetv1
#from lib.nets.vgg16gate import vgg16
from lib.utils.timer import Timer
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from lib.utils.test import test_net
from lib.datasets.factory import get_imdb

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
# CLASSES = ('__background__',
#            'finger')
CLASSES = ('__background__',
            'zangpian','loujiang','duanshan','bengbian')
            #'crackle')
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def vis_detections(det_txt, image_id, ax, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    #print("+_+")
    #print(class_name,dets,thresh)
    inds = np.where(dets[:, -1] >= thresh)[0]
    print("!!!")
    print(inds) # 是否检测出来东西，如果有的话为0如果没有为空
    if len(inds) == 0:
        return


    #print(im.shape)  # 4000 6000 3
    #调整通道顺序，如果不调整通道顺序，图像就不正常
    test_proposal = np.shape(dets)
    num_proposal = test_proposal[0]
    for i in inds:
    #for i in range(num_proposal):
        bbox = dets[i, :4]
        score = dets[i, -1]

        #print(bbox[0],bbox[1],bbox[2],bbox[3])
        print("add one patch")
        det_txt.write(str(image_id) + ' ' + str(score) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(
            bbox[2]) + ' ' + str(bbox[3]))
        det_txt.write("\n")
        if class_name == 'finger':
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=2)
            )
            ax.text(bbox[0]-96, bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=18, color='white')
        else:
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='cyan', linewidth=2)
                        )
            ax.text(bbox[0], bbox[1] - 2,
                            '{:s} {:.3f}'.format(class_name, score),
                            bbox=dict(facecolor='blue', alpha=0.5),
                            fontsize=18, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,thresh),fontsize=12)



def demo(sess, net, image_name, det_txt):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo_bengbian', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()

    timer.tic()
    # detect the picture to find score and boxes
    scores, boxes = im_detect(sess, net,im,image_name)
    # 检测主体部分,在这里加上save_feature_picture
    # 这里的net内容是vgg

    timer.toc()

    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.1

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(im, aspect='equal')
    image_id = image_name.split('.')[0]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= 0.5)[0]
        print("!!!")
        print(inds)  # 是否检测出来东西，如果有的话为0如果没有为空
        if len(inds) == 0:
            a = 1
        else:
            a = 0

        vis_detections(det_txt, image_id, ax,im, cls, dets, thresh=CONF_THRESH)
        # vis_detections(det_txt, image_id, ax, im, cls, dets, thresh=CONF_THRESH)
        plt.draw()
    return a


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='resnetv1')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    # tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])
    # tfmodel=r'D:\Users\Binyi_Su\Experiments\Faster-rcnn\Faster-RCNN-TensorFlow-Python3.5-master\output\vgg16\voc_2007_trainval\default\vgg16_faster_rcnn_iter_20000.ckpt'
    #tfmodel='default18/voc_2007_trainval/default/vgg16_faster_rcnn_iter_40000.ckpt'
    tfmodel = 'default_resnet/voc_2007_trainval/resnetv1-att/vgg16_faster_rcnn_iter_40000.ckpt'
    #tfmodel ='D://Users//lc//Faster - RCNN - lc//default_resnet_att//voc_2007_trainval//resnetv1//vgg16_faster_rcnn_iter_40000.ckpt'
    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta4'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'resnetv1':
        net = resnetv1(batch_size=1, num_layers=50)
    else:
        raise NotImplementedError

    layers = net.create_architecture(sess, "TEST", 5,
                            tag='default', anchor_scales=[8, 16, 32])
    # spatial_attention = tf.reshape(spatial_attention,[32, 32])
    print(layers)
    # plt.imshow(spatial_attention)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # im_names = ['65.bmp', '66.bmp', '67.bmp', '68.bmp',
    #             '69.bmp', '70.bmp','71.bmp','72.bmp','73.bmp','74.bmp','75.bmp']
    # im_names = ['1.jpg', '2.jpg', '3.jpg', '4.jpg',
    #             '5.jpg','6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg']
    im_names = os.listdir('F:\\fastercnn\\data\\demo_bengbian')


    det_txt = open(
       "F:\\fastercnn\\default_resnet\\test-txt\\resnet50_att_bengbian.txt", 'w')
    sum = 0
    total = 0
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))

        a = demo(sess, net, im_name, det_txt)
        total = total + 1
        sum = sum+a
        # demo(sess, net, im_name, det_txt)

        # output RPCAN feature map
        # spatial_attention = np.reshape(spatial_attention, (38, 38))
        # spatial_attention = cv2.resize(spatial_attention,(128,128))
        # plt.imshow(spatial_attention)
        # plt.imsave('C:\\Users\\Administrator\\Desktop\\feature_map\\a.jpg', spatial_attention)
        plt.axis('off')
        plt.savefig("default_resnet/testatt/finish"+im_name)
    # imdb = get_imdb("voc_2007_trainval")
    # test_net(sess, net, imdb, 'default')
      #  plt.show()
    print("sum=",sum)
    print("total=", total)


