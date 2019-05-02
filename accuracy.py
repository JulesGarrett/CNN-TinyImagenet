import model
import cnn
import operator
import softmax as sm
import numpy as np

def feed_forward(img, label, params, conv_s, pool_f, pool_s):

    [f1, f2, f3, f4, f5, w6, w7, b1, b2, b3, b4, b5, b6, b7] = params

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


def accuracy(num_classes = 10, lr = 0.01, beta1 = 0.95, beta2 = 0.99, img_dim = 28, img_depth = 1, f = 2, num_filt1 = 8, num_filt2 = 8, num_filt3 = 8, num_filt4 = 8, num_filt5 = 8, batch_size = 32, num_epochs = 1, save_path = 'test.pkl'):
    # training data
    m =500
    X = model.extract_data('train-images-idx3-ubyte.gz', m, img_dim)
    y_dash = model.extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
    print(y_dash)
    X-= int(np.mean(X))
    X/= int(np.std(X))
    print(X.shape, y_dash.shape)
    train_data = np.hstack((X,y_dash))
    print(train_data.shape)

    np.random.shuffle(train_data)

    f1, f2, f3, f4, f5, w6, w7 = (num_filt1 ,img_depth,f,f), (num_filt2 ,num_filt1,f,f), (num_filt3, num_filt2, f, f), (num_filt4, num_filt3, f, f), (num_filt5, num_filt4, f, f), (128,8), (10, 128)
    f1 = model.initializeFilter(f1)
    f2 = model.initializeFilter(f2)
    f3 = model.initializeFilter(f3)
    f4 = model.initializeFilter(f4)
    f5 = model.initializeFilter(f5)
    w6 = model.initializeWeight(w6)
    w7 = model.initializeWeight(w7)

    b1 = np.zeros((f1.shape[0],1))
    b2 = np.zeros((f2.shape[0],1))
    b3 = np.zeros((f3.shape[0],1))
    b4 = np.zeros((f4.shape[0],1))
    b5 = np.zeros((f5.shape[0],1))
    b6 = np.zeros((w6.shape[0],1))
    b7 = np.zeros((w7.shape[0],1))

    params = [f1, f2, f3, f4, f5, w6, w7, b1, b2, b3, b4, b5, b6, b7]

    cost = []

    print("LR:"+str(lr)+", Batch Size:"+str(batch_size))

    probs = feed_forward(train_data, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
    print(probs)

    maxProb = max(probs)
    index, value = max(enumerate(my_list), key=operator.itemgetter(1))
    print(index)




def main():
    accuracy()

if __name__== "__main__":
  main()
