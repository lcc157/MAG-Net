import os

tfmodel = 'default/voc_2007_trainval/default/vgg16_faster_rcnn_iter_30000.ckpt'
if not os.path.isfile(tfmodel + '.meta'):
    print(tfmodel)
    raise IOError(('{:s} not found.\nDid you download the proper networks from '
                   'our server and place them properly?').format(tfmodel + '.meta'))