# MAG-Net
Tensorflow MAG-Net for Windows by using Python 3.6 

# How To Use This Branch
1- Install tensorflow, preferably GPU version. Follow [instructions]( https://www.tensorflow.org/install/install_windows). If you do not install GPU version, you need to comment out all the GPU calls inside code and replace them with relavent CPU ones.

2- Install python packages (cython, python-opencv, easydict)

3- Checkout this branch

4- Go to  ./data/coco/PythonAPI

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run `python setup.py build_ext --inplace`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Run `python setup.py build_ext install`

5- Follow this instruction to download PyCoco database. [Link]( https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)

I will be glad if you can contribute with a batch script to automatically download and fetch. The final structure has to look like

  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"data/VOCDevkit2007/annotations_cache"
  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"data/VOCDevkit2007/VOC2007"
  
 6- Download pre-trained VGG16 from [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) and place it as "data\imagenet_weights\vgg16.ckpt"
 
 For rest of the models, please check [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
 
  7- Run train.py
  
  Notify me if there is any issue found. Please note that, I have compiled cython modules with sm61 architecture (GTX 1060, 1070 etc.). Compile support for other architectures will be added. 
  
 7- attention
  There are three attentions under the lib folder: RCAG(ours), CBAM, DANET, ECA-NET et al.
 
