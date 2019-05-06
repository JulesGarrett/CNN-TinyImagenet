# from model import initializeFilter
import cnn
import operator
from tqdm import tqdm
import pickle
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


def calculateAccuracy(batch, num_classes, dim, n_c, params, cost):
    X = batch[:,0:-1] # get batch inputs
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:,-1] # get batch labels
    batch_size = len(batch)

    accurate = 0

    # t = tqdm(batch_size)
    for i in range(batch_size):
        print(i)
        # t.set_description("Iteration: %d" % i)
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) # convert label to one-hot

        #Feed forward
        probs = feed_forward(x, y, params, 1, 2, 2)

        index, value = max(enumerate(probs), key=operator.itemgetter(1))
        ind = int(Y[i])

        #Use this to debug:
        # print("\t\nComparing", index, " with actual  value ", ind,"\n")

        if index == ind:
            accurate += 1

    return accurate

def accuracy(num_classes = 10, img_dimen = 64, img_depth = 3, load_params_path = 'test-lr-0-5.pkl'):
    #TRAINING DATA
    with open('wnids.txt') as file: #get relevant 200 idsg
        ids = [line.rstrip('\n') for line in file]
    ids = ids[0:10]

    lines = None
    with open('words.txt') as file:
        lines = [line.rstrip('\n') for line in file]


    label_dict = {}
    id_num = 0
    for line in lines: #map n* id to numbers 0 to 9
        id = line[0:line.index('\t')]
        label = line[line.index('\t')+1:]
        if id not in label_dict and id in ids:
            label_dict[id] = id_num
            id_num += 1
            print(label)

    m =500
    data = np.load('extra-tiny-imagenet.npz')
    X = data['train'].astype(np.float32)[0:2500]

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
    y = y.astype(np.float32)[0:2500]


    num_images = X.shape[0]
    img_len = X.shape[1]
    img_dim = X.shape[-1]
    print("before:", X.shape, y.shape)
    X = X.reshape(num_images,img_len*img_len*img_dim)
    y = y.reshape(num_images,1) # (100000,) -> (100000,1)

    X/= int(np.std(X))
    print("after", X.shape, y.shape)
    train_data = np.hstack((X,y))

    np.random.shuffle(train_data)

    params = pickle.load(open(load_params_path, 'rb'))
    # print("Params:", params)

    cost = []

    accurate = calculateAccuracy(train_data, num_classes, img_dimen, img_depth, params, cost)
    accuracy = accurate/len(train_data) * 100
    print("Accuracy:\n\t", accurate, "/", len(train_data), "\n\t", accuracy, "%")


def main():
    accuracy()

if __name__== "__main__":
  main()
