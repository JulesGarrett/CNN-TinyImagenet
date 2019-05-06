# from model import initializeFilter
import cnn
import operator
import softmax as sm
import numpy as np



def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01



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

    return probs


def calculateAccuracy(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    X = batch[:,0:-1] # get batch inputs
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:,-1] # get batch labels
    batch_size = len(batch)

    accurate = 0

    for i in range(batch_size):
        print(i)
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) # convert label to one-hot

        #Feed forward
        probs = feed_forward(x, y, params, 1, 2, 2)

        index, value = max(enumerate(probs), key=operator.itemgetter(1))
        ind = int(Y[i])

        # print("Comparing", index, " with actual  value ", ind)
        if index == ind:
            accurate += 1

    return accurate

def accuracy(num_classes = 10, lr = 0.01, beta1 = 0.95, beta2 = 0.99, img_dimen = 64, img_depth = 3, f = 2, num_filt1 = 8, num_filt2 = 8, num_filt3 = 8, num_filt4 = 8, num_filt5 = 8, batch_size = 32, num_epochs = 1, save_path = 'test.pkl'):
    # training data
    with open('wnids.txt') as file: #get relevant 200 ids
        ids = [line.rstrip('\n') for line in file]
    ids = ids[0:10]
    #print(ids)

    lines = None
    with open('words.txt') as file:
        lines = [line.rstrip('\n') for line in file]
    #pp.pprint(lines)
    label_dict = {}
    id_num = 0
    for line in lines: #map n* id to numbers 0 to 9
        id = line[0:line.index('\t')]
        label = line[line.index('\t')+1:]
        if id not in label_dict and id in ids:
            label_dict[id] = id_num
            id_num += 1
            print(label)

    #pp.pprint(label_dict)
    #print(label_dict)

    m =500
    data = np.load('extra-tiny-imagenet.npz')
    X = data['train'].astype(np.float32)
    #pp.pprint(X)
    y = []
    temp_y = data['labels']
    print("temp shape",temp_y.shape)
    for label in temp_y:
        if label in ids:
            y.append(label)

    for i in range(len(y)):
        y[i] = label_dict[y[i]]
    y = np.array(y)
    print(y.shape)
    y = y.astype(np.float32)
 #   for i in range(len(y)):
 #       print(y[i])
    num_images = X.shape[0]
    img_len = X.shape[1]
    img_dim = X.shape[-1]
    print("before:", X.shape, y.shape)
    X = X.reshape(num_images,img_len*img_len*img_dim)
    y = y.reshape(num_images,1) # (100000,) -> (100000,1)
    #train_data = np.hstack((X,y))

    # y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
    X-= int(np.mean(X))
    X/= int(np.std(X))
    print("after", X.shape, y.shape)
    train_data = np.hstack((X,y))
    # for i in range(len(train_data)):
    #     print(train_data[:, -1][i])
    # print(train_data.shape)

    np.random.shuffle(train_data)

    f1, f2, f3, f4, f5, w6, w7 = (num_filt1 ,img_depth,f,f), (num_filt2 ,num_filt1,f,f), (num_filt3, num_filt2, f, f), (num_filt4, num_filt3, f, f), (num_filt5, num_filt4, f, f), (128,288), (10, 128)
    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    f3 = initializeFilter(f3)
    f4 = initializeFilter(f4)
    f5 = initializeFilter(f5)
    w6 = initializeWeight(w6)
    w7 = initializeWeight(w7)

    b1 = np.zeros((f1.shape[0],1))
    b2 = np.zeros((f2.shape[0],1))
    b3 = np.zeros((f3.shape[0],1))
    b4 = np.zeros((f4.shape[0],1))
    b5 = np.zeros((f5.shape[0],1))
    b6 = np.zeros((w6.shape[0],1))
    b7 = np.zeros((w7.shape[0],1))

#     params = [f1, f2, f3, f4, f5, w6, w7, b1, b2, b3, b4, b5, b6, b7]
    params = pickle.load(open(save_path, 'rb')) #Obtaining parameters from model.py 's output

    cost = []

    print("LR:"+str(lr)+", Batch Size:"+str(batch_size))
    
    accurate = calculateAccuracy(train_data, num_classes, lr, img_dimen, img_depth, beta1, beta2, params, cost)
    print(accurate)
    accuracy = accurate/len(train_data) * 100
    print("Accuracy: ", accuracy, "%")


def main():
    accuracy()

if __name__== "__main__":
  main()
