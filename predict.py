import mxnet as mx
from mxnet import image
from utils import getTop
from mxnet.gluon.model_zoo.vision import mobilenet1_0
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model", required=True)
    parser.add_argument("-i","--image", required=True)
    parser.add_argument("-g","--gpu", default = False)
       
    args = vars(parser.parse_args())
    
    use_gpu = args["gpu"]
    ctx = mx.gpu() if use_gpu else mx.cpu()
    
    net = mobilenet1_0(classes=14,ctx=ctx)
    net.load_parameters(args["model"],ctx=ctx)
    
    file_path = args["image"]
    
    img = image.imread(file_path).astype('float32')
    getTop(img,net,5)