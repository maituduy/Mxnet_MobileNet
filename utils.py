from mxnet import nd, image

def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()

def eval(data_loader, net, ctx):
    valid_acc = 0.
    for data, label in data_loader:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        valid_acc += acc(net(data), label)
    return valid_acc/len(data_loader)
  
def getTop(data,net,k=1):
    names = ['conservative','dressy','ethnic','fairy','feminine','gal','girlish','kireime-casual','lolita',
         'mode','natural','retro','rock','street']
    data = image.imresize(data,256,256)
    data = image.center_crop(data,(224,224))
    data = nd.transpose(data[0],(2,0,1))/255
    data = nd.expand_dims(data,0)
    result = nd.softmax(net(data))[0]    
    probs = result.sort()[-k:]
    result = result.argsort()[-k:]
    temp = []
    for i in result:
        temp.append(names[int(i.asscalar())])
    for x,y in zip(temp,probs):
        print("name: {}, prob: {:.2f}%".format(x,y.asscalar()*100))

    
    