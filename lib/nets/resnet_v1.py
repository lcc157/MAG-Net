# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import numpy as np
from tflearn.layers.conv import global_avg_pool, global_max_pool
from lib.nets.network import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from lib.config import config as cfg

def resnet_arg_scope(is_training=True,
                    # weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     weight_decay=cfg.FLAGS.weight_decay,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    # NOTE 'is_training' here does not work because inside resnet it gets reset:
    # https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py#L187
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': False,
    'updates_collections': ops.GraphKeys.UPDATE_OPS
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=nn_ops.relu,
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc

class resnetv1(Network):
  def __init__(self, batch_size=1, num_layers=50):
    Network.__init__(self, batch_size=batch_size)
    self._num_layers = num_layers
    self._resnet_scope = 'resnet_v1_%d' % num_layers

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be backpropagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
      if cfg.FLAGS.RESNET_MAX_POOL:
        pre_pool_size = cfg.FLAGS.roi_pooling_size * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")
        crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
      else:
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.FLAGS.roi_pooling_size, cfg.FLAGS.roi_pooling_size],
                                         name="crops")
    return crops

  # Do the first few layers manually, because 'SAME' padding can behave inconsistently
  # for images of different sizes: sometimes 0, sometimes 1
  def build_base(self):
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
      net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

    return net

  def build_network(self, sess, is_training=True):
    # select initializers
    if cfg.FLAGS.initializer == "truncated":
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
    bottleneck = resnet_v1.bottleneck
    # choose different blocks for different number of layers
    if self._num_layers == 50:
      # blocks = [
      #   resnet_utils.Block('block1', bottleneck,
      #                      [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      #   resnet_utils.Block('block2', bottleneck,
      #                      [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      #   # Use stride-1 for the last conv4 layer
      #   resnet_utils.Block('block3', bottleneck,
      #                      [(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
      #   resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      # ]
      blocks = [
          resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
          resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
          resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
          resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
      ]
    elif self._num_layers == 101:
      blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 1)] * 22 + [(1024, 256, 1)]),
        resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      ]
      blocks = [
          resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
          resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
          resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=23, stride=1),
          resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]
    elif self._num_layers == 152:
      blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 1)] * 35 + [(1024, 256, 1)]),
        resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
      ]
    else:
      # other numbers are not supported
      raise NotImplementedError

    assert (0 <= cfg.FLAGS.RESNET_FIXED_BLOCKS < 4)
    if cfg.FLAGS.RESNET_FIXED_BLOCKS == 3:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[0:cfg.FLAGS.RESNET_FIXED_BLOCKS],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    elif cfg.FLAGS.RESNET_FIXED_BLOCKS > 0:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net, _ = resnet_v1.resnet_v1(net,
                                     blocks[0:cfg.FLAGS.RESNET_FIXED_BLOCKS],
                                     global_pool=False,
                                     include_root_block=False,
                                     scope=self._resnet_scope)

      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[cfg.FLAGS.RESNET_FIXED_BLOCKS:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    else:  # cfg.RESNET.FIXED_BLOCKS == 0
      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net = self.build_base()
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[0:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
    #net_conv4 = convolutional_block_attention_module_SC(net_conv4, 1)

    self._act_summaries.append(net_conv4)
    self._layers['head'] = net_conv4
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # build the anchors for the image
      self._anchor_component()

      # rpn
      rpn = slim.conv2d(net_conv4, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                         scope="rpn_conv/3x3")
      #rpn_first=slim.conv2d(net_conv4, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                     #   scope="rpn_conv/3x3")
     # rpn=convolutional_block_attention_module_SC(rpn_first,1)
     # self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')
      # change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
      if is_training:
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      else:
        if cfg.FLAGS.test_mode == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.FLAGS.test_mode == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError

      # rcnn
    #pool5 = self._crop_pool_layer(net_conv4, rois, "pool5")


    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
      pool5 = self._crop_pool_layer(net_conv4, rois, "pool5")
      fc7, _ = resnet_v1.resnet_v1(pool5,
                                   blocks[-1:],
                                   global_pool=False,
                                   include_root_block=False,
                                   scope=self._resnet_scope)

    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # Average pooling done by reduce_mean
      fc7 = tf.reduce_mean(fc7, axis=[1, 2])
      cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer,
                                       trainable=is_training, activation_fn=None, scope='cls_score')
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
                                       trainable=is_training,
                                       activation_fn=None, scope='bbox_pred')
    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["cls_score"] = cls_score
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred
    self._predictions["rois"] = rois

    self._score_summaries.update(self._predictions)

    return rois, cls_prob, bbox_pred

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._resnet_scope + '/conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Varibles restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix Resnet V1 layers..')
    with tf.variable_scope('Fix_Resnet_V1') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR
        conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._resnet_scope + "/conv1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                           tf.reverse(conv1_rgb, [2])))




def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape


def PAM_Module(in_dim):
    """ Position attention module"""
    # Ref from SAGAN
    #in_dim1 = combined_static_and_dynamic_shape(in_dim)
    chanel_in = in_dim
    query_conv = slim.conv2d(in_dim, 128, kernel_size=1, padding='VALID')
    key_conv = slim.conv2d(in_dim, 128, kernel_size=1, padding='VALID')
    value_conv = slim.conv2d(in_dim, 128, kernel_size=1, padding='VALID')
    gamma = tf.Variable(tf.zeros([1]),name='gamma')
   # softmax = tf.nn.softmax(dim=-1)
    m_batchsize,  height, width ,C =combined_static_and_dynamic_shape(chanel_in)

    #proj_query = query_conv(chanel_in).view(m_batchsize, -1, width * height).permute(0, 2, 1)
    proj_query = tf.reshape(query_conv,[m_batchsize, -1, width * height])
    proj_query = tf.transpose(proj_query, [0, 2, 1])
   # proj_key = key_conv(chanel_in).view(m_batchsize, -1, width * height)
    proj_key = tf.reshape(key_conv,[m_batchsize, -1, width * height])
    energy = tf.matmul(proj_query, proj_key)  # 矩阵乘法
    attention = tf.nn.softmax(energy,dim=-1)  # 添加非线性函数
    #proj_value = value_conv(chanel_in).view(m_batchsize, -1, width * height
    proj_value = tf.reshape(value_conv, [m_batchsize, -1, width * height])
    attention = tf.transpose(attention, [0, 2, 1])
    out = tf.matmul(proj_value, attention)
    out = tf.reshape(out, [m_batchsize,  height, width ,C])  # reshape到原图
   # out =gamma * out
    return out
def NCAM_Module(in_dim):
    """ Position attention module"""
    # Ref from SAGAN
    chanel_in = in_dim
    #gamma = tf.Variable(tf.zeros([1]),name='gamma')
    m_batchsize,  height, width ,C =combined_static_and_dynamic_shape(chanel_in)
    globel_avg = global_avg_pool(chanel_in)
    channel_avg_weights = tf.reshape(globel_avg, [1, C, -1])
    globel_max = global_max_pool(chanel_in)
    channel_max_weights = tf.reshape(globel_max, [1, -1, C])
    energy = tf.matmul(channel_avg_weights, channel_max_weights)  # 矩阵乘法
    attention = tf.nn.softmax(energy,dim=-1)  # 添加非线性函数
    proj_value_CAM = tf.reshape(chanel_in, [m_batchsize, C, -1])
    out = tf.matmul(attention , proj_value_CAM)
    out = tf.reshape(out, [m_batchsize,  height, width ,C])  # reshape到原图
  #  out =gamma * out
    out = PAM_Module(out)
    out = out+chanel_in
    return out
def CAM1_Module(in_dim):
    """ Position attention module"""
    # Ref from SAGAN
    chanel_in = in_dim
    gamma = tf.Variable(tf.zeros([1]),name='gamma')
    m_batchsize,  height, width ,C =combined_static_and_dynamic_shape(chanel_in)
    proj_query_CAM = tf.reshape(chanel_in,[m_batchsize, C, -1])
    proj_key_CAM = tf.reshape(chanel_in,[m_batchsize, C, -1])
    proj_key_CAM = tf.transpose(proj_key_CAM, [0, 2, 1])
    energy = tf.matmul(proj_query_CAM, proj_key_CAM)  # 矩阵乘法
    #energy_new = tf.reduce_max(energy,-1,keepdims=True)
    # #energy_new = energy_new[0].expand_as(energy)
    #energy_new = np.expand_dims(energy_new,axis=-1)
    #energy_new = energy_new - energy
    attention = tf.nn.softmax(energy)  # 添加非线性函数
    proj_value_CAM = tf.reshape(chanel_in, [m_batchsize, C, -1])
    out = tf.matmul(attention , proj_value_CAM)
    out = tf.reshape(out, [m_batchsize,  height, width ,C])  # reshape到原图
    out =gamma * out
    #out = PAM_Module(out)
    out = out+chanel_in
    return out
def aspp(inputs, output_stride, batch_norm_decay, is_training, depth=256):
    '''实现ASPP
    参数：
      inputs：输入四维向量
      output_stride：决定空洞卷积膨胀率
      batch_norm_decay:同上函数
      is_training:是否训练
      depth:输出通道数
    返回值：
      ASPP后的输出
      '''
    with tf.variable_scope('aspp'):
        if output_stride not in [8, 16]:
            raise ValueError('out_stride整错了')
        # 膨胀率
       # atrous_rates = [6, 12, 18]
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=0.0005)):
            with slim.arg_scope([slim.conv2d],
                                weights_initializer=slim.xavier_initializer(),

                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': batch_norm_decay}):
                inputs_size = tf.shape(inputs)[1:3]
                # slim.conv2d默认激活函数为relu,padding=SAME
                conv_1x1 = slim.conv2d(inputs, depth, [1, 1], stride=1, scope='conv_1x1')
                # 空洞卷积rate不为1
                conv_3x3_1 = slim.conv2d(inputs, depth, [3, 3], stride=1, rate=1, scope='conv_3x3_1')
                conv_3x3_2 = slim.conv2d(inputs, depth, [3, 3], stride=1, rate=2, scope='conv_3x3_2')
                conv_3x3_3 = slim.conv2d(inputs, depth, [3, 3], stride=1, rate=4, scope='conv_3x3_3')
               # pcam = PAM_Module(inputs)
                with tf.variable_scope('image_level_features'):
                     # 池化
                     image_level_features = tf.reduce_mean(inputs, axis=[1, 2], keep_dims=True,
                                                          name='global_average_pooling')
                     image_level_features = slim.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
                #     # 双线性插值
                     image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
                net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3,
                                name='concat')
                net = slim.conv2d(net,512,[1,1],trainable=is_training, scope='convq')
                return net
def PCAM_Module(in_dim):
    #---------------------------------------------------pam----------------------------
    chanel_in = in_dim
    query_conv = slim.conv2d(in_dim, 512, kernel_size=1, padding='VALID', scope='query_conv')
    key_conv = slim.conv2d(in_dim, 512, kernel_size=1, padding='VALID', scope='key_conv')
    value_conv = slim.conv2d(in_dim, 512, kernel_size=1, padding='VALID', scope='value_conv')
    gamma = tf.Variable(tf.zeros([1]), name='gamma')
    camma = tf.Variable(tf.zeros([1]), name='camma')
    m_batchsize, height, width, C = combined_static_and_dynamic_shape(chanel_in)
    proj_query = tf.reshape(query_conv, [m_batchsize, -1, width * height])
    proj_query = tf.transpose(proj_query, [0, 2, 1])
    # proj_key = key_conv(chanel_in).view(m_batchsize, -1, width * height)
    proj_key = tf.reshape(key_conv, [m_batchsize, -1, width * height])
    energy = tf.matmul(proj_query, proj_key)  # 矩阵乘法
    attention = tf.nn.softmax(energy)  # 添加非线性函数
    # proj_value = value_conv(chanel_in).view(m_batchsize, -1, width * height
    proj_value = tf.reshape(value_conv, [m_batchsize, -1, width * height])
    attention = tf.transpose(attention, [0, 2, 1])
    out_PAM = tf.matmul(proj_value, attention)
    out_PAM = tf.reshape(out_PAM, [m_batchsize, height, width, C])  # reshape到原图
    out_PAM = gamma * out_PAM + chanel_in
    # ----------------------------------------------------cam----------------------------
    proj_query_CAM = tf.reshape(chanel_in, [m_batchsize, C, -1])
    proj_key_CAM = tf.reshape(chanel_in, [m_batchsize, C, -1])
    proj_key_CAM = tf.transpose(proj_key_CAM, [0, 2, 1])
    energy_CAM = tf.matmul(proj_query_CAM, proj_key_CAM)  # 矩阵乘法
    # energy_new = tf.reduce_max(energy_CAM, -1, keepdim=True)
    # energy_new = energy_new[0].expand_as(energy_CAM)
    # energy_new = energy_new - energy_CAM
    attention_CAM = tf.nn.softmax(energy_CAM)  # 添加非线性函数
    proj_value_CAM = tf.reshape(chanel_in, [m_batchsize, C, -1])
    out_CAM = tf.matmul(attention_CAM, proj_value_CAM)
    out_CAM = tf.reshape(out_CAM, [m_batchsize, height, width, C])  # reshape到原图
    out_CAM = camma * out_CAM + chanel_in
    out =out_CAM+out_PAM

    return out
def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape
def convolutional_block_attention_module(feature_map, index, inner_units_ratio=0.5):
    """
    CBAM: convolution block attention module, which is described in "CBAM: Convolutional Block Attention Module"
    Architecture : "https://arxiv.org/pdf/1807.06521.pdf"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention
    """
    with tf.variable_scope("cbam_%s" % (index)):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # feature_map_shape[0] = 1
        # feature_map_shape[1] = 32
        # feature_map_shape[2] = 32
        # feature_map_shape[3] = 512
        # channel attention

        # channel_avg_weights = tf.nn.avg_pool(
        #     value=feature_map,
        #     ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        #     strides=[1, 1, 1, 1],
        #     padding='VALID'
        # )
        # channel_max_weights = tf.nn.max_pool(
        #     value=feature_map,
        #     ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        #     strides=[1, 1, 1, 1],
        #     padding='VALID'
        # )
        globel_avg = global_avg_pool(feature_map)
        channel_avg_weights = tf.reshape(globel_avg, [1, 1, 1, 512])
        globel_max = global_max_pool(feature_map)
        channel_max_weights = tf.reshape(globel_max, [1, 1, 1, 512])

        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        # channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)
        #
        # fc_1 = tf.layers.dense(
        #     inputs=channel_w_reshape,
        #     units=feature_map_shape[3] * inner_units_ratio,
        #     name="fc_1",
        #     activation=tf.nn.relu
        # )
        # fc_2 = tf.layers.dense(
        #     inputs=fc_1,
        #     units=feature_map_shape[3],
        #     name="fc_2",
        #     activation=None
        # )
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=3)

        fc_1 = slim.conv2d(
            channel_w_reshape,
            feature_map_shape[3] * inner_units_ratio,
            [1, 1],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="channel_attention_conv1"
        )
        fc_2 = slim.conv2d(
            fc_1,
            feature_map_shape[3],
            [1, 1],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="channel_attention_conv2"
        )
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)
        # spatial attention
        channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)

        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2], 1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2], 1])

        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [7, 7],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
        return feature_map_with_attention
        # return feature_map_with_channel_attention
def convolutional_block_attention_module_SC(feature_map, index, inner_units_ratio=0.5):
    """
    CBAM: convolution block attention module, which is described in "CBAM: Convolutional Block Attention Module"
    Architecture : "https://arxiv.org/pdf/1807.06521.pdf"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention
    """
    with tf.variable_scope("cbam_%s" % (index)):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)

        # //spatial attention//
        channel_wise_avg_pooling = tf.reduce_mean(feature_map, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map, axis=3)

        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])

        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        spatial_attention = slim.conv2d(
            channel_wise_pooling,
            1,
            [7, 7],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="spatial_attention_conv"
        )
        feature_map_with_spatial_attention = tf.multiply(feature_map, spatial_attention)
        # //channel attention//
        # channel_avg_weights = tf.nn.avg_pool(
        #     value=feature_map,
        #     ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        #     strides=[1, 1, 1, 1],
        #     padding='VALID'
        # )
        # channel_max_weights = tf.nn.max_pool(
        #     value=feature_map,
        #     ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        #     strides=[1, 1, 1, 1],
        #     padding='VALID'
        # )
        globel_avg = global_avg_pool(feature_map_with_spatial_attention)
        channel_avg_weights = tf.reshape(globel_avg, [1, 1, 1, 512])
        globel_max = global_max_pool(feature_map_with_spatial_attention)
        channel_max_weights = tf.reshape(globel_max, [1, 1, 1, 512])

        # //original program//
        # channel_avg_reshape = tf.reshape(channel_avg_weights,
        #                                  [feature_map_shape[0], 1, feature_map_shape[3]])
        # channel_max_reshape = tf.reshape(channel_max_weights,
        #                                  [feature_map_shape[0], 1, feature_map_shape[3]])
        # channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)
        #
        # fc_1 = tf.layers.dense(
        #     inputs=channel_w_reshape,
        #     units=feature_map_shape[3] * inner_units_ratio,
        #     name="fc_1",
        #     activation=tf.nn.relu
        # )
        # fc_2 = tf.layers.dense(
        #     inputs=fc_1,
        #     units=feature_map_shape[3],
        #     name="fc_2",
        #     activation=None
        # )

        # //improved program//
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, 1, feature_map_shape[3]])
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=3)

        fc_1 = slim.conv2d(
            channel_w_reshape,
            feature_map_shape[3] * inner_units_ratio,
            [1, 1],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="channel_attention_conv1"
        )
        fc_2 = slim.conv2d(
            fc_1,
            feature_map_shape[3],
            [1, 1],
            padding='SAME',
            activation_fn=tf.nn.sigmoid,
            scope="channel_attention_conv2"
        )

        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        feature_map_with_channel_attention = tf.multiply(feature_map_with_spatial_attention, channel_attention)

        return feature_map_with_channel_attention