import mxnet as mx
from mxnet import init
from mxnet.gluon.data.vision import transforms, ImageRecordDataset
from mxnet.gluon.model_zoo.vision import mobilenet1_0
from mxnet import nd, gluon, autograd, image

from multiprocessing import cpu_count

from utils import *
import time 

batch_size = 128

def getNet(ctx):
    pretrained_net = mobilenet1_0(pretrained=True,ctx=ctx)
    net = mobilenet1_0(classes=14,ctx=ctx)
    net.features = pretrained_net.features
    net.output.initialize(init.Xavier(),ctx=ctx)
    
    return net

def getTransform():
    transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(),
    transforms.RandomHue(.1),
    transforms.ToTensor(),
    #     transforms.Normalize([0.64026263, 0.60487159, 0.58312368], [0.23318863, 0.23745599, 0.23655162])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    #     transforms.Normalize([0.64026263, 0.60487159, 0.58312368], [0.23318863, 0.23745599, 0.23655162])
    ])
    
    return transform_train,transform_test

def getData(train_rec,test_rec,validation_rec,batch_size):
    transform_train,transform_test = getTransform()
    
    trainIterator = ImageRecordDataset(filename=train_rec).transform_first(transform_train)
    validationIterator = ImageRecordDataset(filename=validation_rec).transform_first(transform_test)
    testIterator = ImageRecordDataset(filename=test_rec).transform_first(transform_test)

    
    CPU_COUNT = cpu_count()
    # print(CPU_COUNT)
    train_loader = mx.gluon.data.DataLoader(trainIterator, batch_size=batch_size, num_workers=CPU_COUNT, shuffle=True)
    test_loader = mx.gluon.data.DataLoader(testIterator, batch_size=batch_size, num_workers=CPU_COUNT)
    val_loader =mx.gluon.data.DataLoader(validationIterator, batch_size=batch_size, num_workers=CPU_COUNT)
    
    return train_loader,test_loader,val_loader

def train(net,ctx,epochs=30,batch_size=128):
    train_loader,test_loader,val_loader = getData(train_rec,test_rec,validation_rec,batch_size)
    
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001,'momentum': 0.9})

    for epoch in range(epochs):
        train_loss, train_acc, valid_acc = 0., 0., 0.
        tic = time.time()
        for i, (data, label) in enumerate(train_loader):
            st = time.time() 
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # forward + backward
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            # update parameters
            trainer.step(batch_size)

            # calculate training metrics
            temp_loss = loss.mean().asscalar()
            temp_acc = acc(output, label)
            train_loss += temp_loss
            train_acc += temp_acc

    #         print('[Epoch %d Batch %d] speed: %f samples/s train_acc: %f train_loss: %f'%(epoch, i, batch_size/(time.time()-st),temp_acc,temp_loss))

        # calculate validation accuracy
        valid_acc = eval(val_loader,net,ctx)

        print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
                epoch+1, train_loss/len(train_loader), train_acc/len(train_loader),
                valid_acc, time.time()-tic))
    
    return net

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", default=30)
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("-g","--gpu", default = False)
       
    args = vars(parser.parse_args())
    
    train_rec = args["data"]+"/train_rec.rec"
    validation_rec = args["data"]+"/val_rec.rec"
    test_rec = args["data"]+"/test_rec.rec"
    
    use_gpu = args["gpu"]
    ctx = mx.gpu() if use_gpu else mx.cpu()
    
    net = getNet(ctx)
    net = train(net,ctx,30,batch_size)
    net.save_parameters(args["target_model"])
