# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.framework.python.ops import arg_scope
from tflearn.layers.conv import global_avg_pool, global_max_pool
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import lib.config.config as cfg
from lib.nets.network import Network
from keras.layers import Concatenate, Conv2D, Add
import keras

# def get session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# keras .backend.tensorflow_backend.set_session(get_session())
class vgg16(Network):
    def __init__(self, batch_size=1):
        Network.__init__(self, batch_size=batch_size)

    def build_network(self, sess, is_training=True):
        with tf.variable_scope('vgg_16', 'vgg_16'):

            # select initializer
            if cfg.FLAGS.initializer == "truncated":
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            # Build head
            net = self.build_head(is_training)

            # Build rpn
            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape = self.build_rpn(net, is_training, initializer)

            # Build proposals
            rois = self.build_proposals(is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)

            # Build predictions
            cls_score, cls_prob, bbox_pred = self.build_predictions(net, rois, is_training, initializer, initializer_bbox)

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
            # exclude the conv weights that are fc weights in vgg16
            if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == 'vgg_16/conv1/conv1_1/weights:0':
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16'):
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv,
                                              "vgg_16/fc7/weights": fc7_conv,
                                              "vgg_16/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc6/weights:0'], tf.reshape(fc6_conv,
                                                                                              self._variables_to_fix['vgg_16/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/fc7/weights:0'], tf.reshape(fc7_conv,
                                                                                              self._variables_to_fix['vgg_16/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))

    def build_head(self, is_training):

        # Main network
        # Layer  1
        # Layer  1
        net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
        # Layer 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
        # Layer 3
        net = slim.conv2d(net, 256, [3, 3], trainable=is_training, scope='conv3/conv3_1')
        #self._layers['3_1'] = net
        net_3_1 = slim.conv2d(net, 128, [1, 1], padding='VALID', activation_fn=None,trainable=is_training, scope='conv3_1_1')
        net = slim.conv2d(net, 256, [3, 3], trainable=is_training, scope='conv3/conv3_2')
        #self._layers['3_2'] = net
        net_3_2 = slim.conv2d(net, 128, [1, 1], padding='VALID', activation_fn=None,trainable=is_training, scope='conv3_2_2')
        net = slim.conv2d(net, 256, [3, 3], trainable=is_training, scope='conv3/conv3_3')
       # self._layers['3_3'] = net
        net_3_3 = slim.conv2d(net, 128, [1, 1], padding='VALID', activation_fn=None,trainable=is_training, scope='conv3_3_3')

        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
        # Layer 4
        net = slim.conv2d(net, 512, [3, 3], trainable=is_training, scope='conv4/conv4_1')
       # self._layers['4_1'] = net
        net_4_1 = slim.conv2d(net, 128, [1, 1], padding='VALID', activation_fn=None,trainable=is_training, scope='conv4_1_1')
        net = slim.conv2d(net, 512, [3, 3], trainable=is_training, scope='conv4/conv4_2')
       # self._layers['4_2'] = net
        net_4_2 = slim.conv2d(net, 128, [1, 1], padding='VALID', activation_fn=None,trainable=is_training, scope='conv4_2_2')
        net = slim.conv2d(net, 512, [3, 3], trainable=is_training, scope='conv4/conv4_3')
       # self._layers['4_3'] = net
        net_4_3 = slim.conv2d(net, 128, [1, 1], padding='VALID',activation_fn=None, trainable=is_training, scope='conv4_3_3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
        # Layer 5
        net = slim.conv2d(net, 512, [3, 3], trainable=is_training, scope='conv5/conv5_1')
       # self._layers['5_1'] = net
        net_5_1 = slim.conv2d(net, 128, [1, 1], padding='VALID', activation_fn=None,trainable=is_training, scope='conv5_1_1')
        net = slim.conv2d(net, 512, [3, 3], trainable=is_training, scope='conv5/conv5_2')
       # self._layers['5_2'] = net
        net_5_2 = slim.conv2d(net, 128, [1, 1], padding='VALID',activation_fn=None, trainable=is_training, scope='conv5_2_1')
        net = slim.conv2d(net, 512, [3, 3], trainable=is_training, scope='conv5/conv5_3')
       # self._layers['5_3'] = net
        # SLIDE Lyer 5
        net_5_3 = slim.conv2d(net, 128, [1, 1], padding='VALID',activation_fn=None, trainable=is_training, scope='conv5_3_1')
        net_5_add = net_5_1 + net_5_2 + net_5_3
        net_5_add = slim.conv2d(net_5_add, 256, [3, 3], padding='SAME',activation_fn=None, trainable=is_training, scope='conv5_3_1_ADD')
        net_5_add = tf.nn.relu(net_5_add)
        net_5_add = NCAM_Module(net_5_add)#1x1-3x
        # SLIDE Lyer 4
        net_4_add = net_4_1 + net_4_2 + net_4_3
        inputs_size_net4 = tf.shape(net_4_add)[1:3]
        net_4_up = tf.image.resize_bilinear(net_5_add, inputs_size_net4, name='upsample')
        net_4_add = net_4_add + net_4_up
        net_4_add = slim.conv2d(net_4_add, 256, [1, 1], padding='SAME', activation_fn=None,scope='net_4_add')
        net_4_add = slim.conv2d(net_4_add, 256, [3, 3], stride=2, padding='SAME', activation_fn=None,scope='net_4_2add')
        net_4_add = tf.nn.relu(net_4_add)
        net_4_add = NCAM_Module(net_4_add)
        # SLIDE Lyer 3
        net_3_add = net_3_1 + net_3_2 + net_3_3
        inputs_size_net3 = tf.shape(net_3_add)[1:3]
        net_3_up = tf.image.resize_bilinear(net_4_add, inputs_size_net3, name='upsample')
        net_3_add = net_3_add + net_3_up
        net_3_add = slim.conv2d(net_3_add, 256, [1, 1], padding='SAME',activation_fn=None, scope='net_3_add')
        net_3_add = slim.repeat(net_3_add, 2, slim.conv2d, 256, [3, 3], stride=2, padding='SAME', activation_fn=None,scope='net_3_1add')
        net_3_add = tf.nn.relu(net_3_add)
        net_3_add = NCAM_Module(net_3_add)
        # NEW fuse
        fuse = tf.concat([net_3_add,net_4_add,net_5_add], axis=3,name='concat')
        net = slim.conv2d(fuse,512,[1,1],padding='VALID',scope='fuse_1')
        net = slim.conv2d(net, 512, [3, 3], padding='SAME', scope='fuse_2')
        net = tf.layers.batch_normalization(net)
        self._act_summaries.append(net)
        # Append network as head layer
        self._layers['head'] = net

        return net

    def build_rpn(self, net, is_training, initializer):

        # Build anchor component
        self._anchor_component()

        # Create RPN Layer
        #rpn_first = slim.conv2d(net, 512, [1, 1], trainable=is_training, weights_initializer=initializer, scope="rpn_conv/3x3")
        #self._dd['rpn'] = rpn_first
       # self._act_summaries.append(rpn_first)
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                                 scope="rpn_conv/3x3")
       # self._act_summaries.append(rpn)
        # # # //spatial attention//
        # # spatial_attention_map = slim.conv2d(rpn_first, 1, [1, 1], trainable=is_training,
        # #                                     weights_initializer=initializer, scope="spatial_attention_map")
        # # rpn = tf.multiply(rpn_first, spatial_attention_map)
        #
        # #//channel_wise attention//
        # channel_wise_averagepooling = torch.nn.AdaptiveAvgPool2d(rpn_first, 1)

        # channel_wise_averagepooling = global_avg_pool(rpn_first)
        # channel_wise_reshape = tf.reshape(channel_wise_averagepooling,[1, 1, 1, 512])
        # # gap_average = tf.layers.average_pooling2d(inputs=rpn_first, pool_size=[32, 32], strides=32)
        # channel_attention_conv_average = slim.conv2d(channel_wise_reshape, 512/16, [1, 1], trainable=is_training, weights_initializer=initializer, scope="channel_attention_conv1")
        # channel_attention_map = slim.conv2d(channel_attention_conv_average, 512, [1, 1], trainable=is_training, weights_initializer=initializer, scope="channel_attention_map")
        # rpn = tf.multiply(rpn_first, channel_attention_map)

        # # spatial_attention = slim.conv2d(rpn_channel, 64, [1, 1], trainable=is_training,
        # #                                     weights_initializer=initializer, scope="spatial_attention")
        # spatial_attention_map = slim.conv2d(rpn_channel, 1, [1, 1], trainable=is_training,
        #                                     weights_initializer=initializer, scope="spatial_attention_map")
        # rpn = tf.multiply(rpn_first, spatial_attention_map)
       # rpn = convolutional_block_attention_module_SC(rpn_first, 1)
       # rpn = PCAM_Module(rpn_first)
     #   self._dd['pcam'] = rpn
     #   self._act_summaries.append(rpn)
        # rpn = tf.multiply(rpn_first, feature_map)

        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_cls_score')

        # Change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn,self._num_anchors * 4, [1, 1], trainable=is_training, weights_initializer=initializer, padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape

    def build_proposals(self, is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):

        if is_training:
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")

            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
        else:
            if cfg.FLAGS.test_mode == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.FLAGS.test_mode == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError
        return rois

    def build_predictions(self, net, rois, is_training, initializer, initializer_bbox):

        # Crop image ROIs
        pool5 = self._crop_pool_layer(net, rois, "pool5")
        pool5_flat = slim.flatten(pool5, scope='flatten')

        # Fully connected layers
        fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
        if is_training:
            fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')

        fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
        if is_training:
            fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')

        # Scores and predictions
        cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer, trainable=is_training, activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        bbox_prediction = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox, trainable=is_training, activation_fn=None, scope='bbox_pred')

        return cls_score, cls_prob, bbox_prediction


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
    attention = tf.nn.softmax(energy)  # 添加非线性函数
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
    attention = tf.nn.softmax(energy)  # 添加非线性函数
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