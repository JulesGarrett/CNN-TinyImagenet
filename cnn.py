import numpy as np
import sys
import pprint as pp
import math


##############################################################
################CONVOLUTIONAL FUNCTIONS#######################
##############################################################

def conv(img, conv_filter, bias, stride=2):
    (n_filt, n_filt_chan, filt, _) = conv_filter.shape
    n_chan, img_dim, _ = img.shape

    out_dim = int((img_dim - filt)/stride) + 1 #calculate output dim
    assert n_chan == n_filt_chan, "filter and image must have same number of channels"

    out = np.zeros((n_filt, out_dim, out_dim))

    #convolve each filter over the image
    for curr_filt in range(n_filt):
        curr_y = out_y = 0
        while curr_y + filt < img_dim:
            curr_x = out_x = 0
            while curr_x +filt <= img_dim:
                out[curr_filt, out_y, out_x] = np.sum(conv_filter[curr_filt] * img[:,curr_y:curr_y+filt, curr_x:curr_x+filt]) + bias[curr_filt]
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
    return out

def conv_back(dconv_prev, conv_in, conv_filter, stride):
    (n_filt, n_filt_chan, filt, _) = conv_filter.shape
    (_, orig_dim, _) = conv_in.shape

    dout = np.zeros(conv_in.shape)
    dfilt = np.zeros(conv_filter.shape)
    dbias = np.zeros((n_filt, 1))
    for curr_filt in range(n_filt):
        curr_y = out_y = 0
        while curr_y + filt <= orig_dim:
            curr_x = out_x = 0
            while curr_x +filt <= orig_dim:
                dfilt[curr_filt] += dconv_prev[curr_filt, out_y, out_x] * conv_in[:, curr_y:curr_y+filt, curr_x:curr_x+filt]
                dout[:, curr_y:curr_y+filt, curr_x:curr_x+filt] += dconv_prev[curr_filt, out_y, out_x] * conv_filter[curr_filt]
                curr_x += stride
                out_x += 1
            curr_y +=stride
            out_y += 1
        dbias[curr_filt] = np.sum(dconv_prev[curr_filt])
    return dout, dfilt, dbias


##############################################################
###################POOLING FUNCTIONS##########################
##############################################################

def pooling(feature_map, size=2, stride=2):
    #Preparing the output of the pooling operation.
    pool_out = np.zeros((np.uint16((feature_map.shape[0]-size+1)/stride+1),
                            np.uint16((feature_map.shape[1]-size+1)/stride+1),
                            feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0,feature_map.shape[0]-size+1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1]-size+1, stride):
                pool_out[r2, c2, map_num] = np.max([feature_map[r:r+size,  c:c+size, map_num]])
                c2 = c2 + 1
            r2 = r2 +1
    return pool_out

#util for pool_back
def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs

def pool_back(dpool, orig, filt, stride):
    (n_chan, orig_dim, _) = orig.shape
    dout = np.zeros(orig.shape)

    for curr_c in range(n_chan):
        curr_y = out_y = 0
        while curr_y + filt <= orig_dim:
            curr_x = out_x = 0
            while curr_x +filt <= orig_dim:
                (a, b) = nanargmax(orig[curr_c, curr_y : curr_y +filt, curr_x:curr_x+filt])
                dout[curr_c, curr_y+a, curr_x +b] = dpool[curr_c, out_y, out_x]

                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
    return dout



def relu(feature_map):
    return feature_map * (feature_map > 0)

#expects weights shape as (activation depth) x (volume of feature map)
def fc(feature_map,weights):
    if (np.prod(feature_map.shape) != weights.shape[-1]):
        print("Number of weights in FC doesn't match volume of feature map.")
        sys.exit()
    #Unpack feature map and return activation layer
    return np.dot(feature_map.reshape(-1),weights.T)


def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions))/N
    return ce
