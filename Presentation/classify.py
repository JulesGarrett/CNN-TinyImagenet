# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'C:\\Users\\Admin\\Documents\\GitHub\\CNN-TinyImagenet')
import cnn
import pickle
import softmax as sm
from os import path



def classify(img, label, params, conv_s, pool_f, pool_s):
    
    params = []
    with (open('C:\\Users\\Admin\\Documents\\GitHub\\CNN-TinyImagenet\\Presentation\\test.pkl', "rb")) as openfile:
        while True:
            try:
                params.append(pickle.load(openfile))
            except EOFError:
                break

    

    [f1, f2, f3, f4, f5, w6, w7, b1, b2, b3, b4, b5, b6, b7] = params[0]

    #forward operations

    conv1 = cnn.conv(img, f1, b1, conv_s)
    conv1 = cnn.relu(conv1)

    pooled1 = sm.maxpool(conv1, pool_f, pool_s)

    conv2 = cnn.conv(pooled1, f2, b2, conv_s)
    conv2 = cnn.relu(conv2)

    pooled2 = sm.maxpool(conv2, pool_f, pool_s)
    # print("pooled2", pooled2.shape)

    conv3 = cnn.conv(pooled2, f3, b3, conv_s)
    conv3 = cnn.relu(conv3)

    conv4 = cnn.conv(conv3, f4, b4, conv_s)
    conv4 = cnn.relu(conv4)

    conv5 = cnn.conv(conv4, f5, b5, conv_s)
    conv5 = cnn.relu(conv5)
    # print("conv5: ", conv5.shape)
    pooled3 = sm.maxpool(conv5, pool_f, pool_s)

    (nf2, dim2, _) = pooled3.shape
    # print("params: ",nf2, dim2)
    fc = pooled3.reshape((nf2*dim2*dim2, 1))

    # print(fc.shape, w6.shape, b6.shape)
    z = w6.dot(fc) + b6
    z = cnn.relu(z)
    out = w7.dot(z) + b7

    probs = sm.softmax(out)
    
    print(probs)
    
    return probs

