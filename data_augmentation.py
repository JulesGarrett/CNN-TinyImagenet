import cv2
import numpy as np
import imageio
from matplotlib import pyplot as plt

'''
    1. 227 or 224?
    2. channel first or last?
'''
IMAGE_SIZE = 64
CROP_SIZE = 58
'''
    @input: any image with size equal to or larger than 256,
    @return: a list of 2^4 images of the size 224x224
'''
#resize the image to 256x256
def resize_image(img):
    resized_img = cv2.resize(img,(IMAGE_SIZE, IMAGE_SIZE))
    return resized_img

#flip image
def flip_image(img):
    flipped_image = cv2.flip(img, 0)
    return flipped_image

#crop 2^3 images from input image
def crop_image(img):
    imgs = []
    diff = IMAGE_SIZE - CROP_SIZE
    imgs.append(img[0:CROP_SIZE,0:CROP_SIZE])
    imgs.append(img[1:CROP_SIZE+1,0:CROP_SIZE ])
    imgs.append(img[diff:IMAGE_SIZE,0:CROP_SIZE])
    imgs.append(img[diff-1:IMAGE_SIZE-1,0:CROP_SIZE])
    imgs.append(img[0:CROP_SIZE,diff-1:IMAGE_SIZE-1])
    imgs.append(img[0:CROP_SIZE,diff:IMAGE_SIZE])
    imgs.append(img[diff:IMAGE_SIZE,diff:IMAGE_SIZE])
    imgs.append(img[diff:IMAGE_SIZE,diff-1:IMAGE_SIZE-1])
#    for i in imgs:
#        plt.imshow(i)
#        plt.show()
    return imgs


def data_aug(img):


    resized_img = resize_image(img)
#    cv2.imshow('resized image', resized_img)

    flipped_img = np.flip(resized_img, axis= 0)
#    plt.imshow(resized_img, interpolation='nearest')

#    plt.imshow(flipped_img, interpolation='nearest')
#    plt.show()


    cropped_img = crop_image(resized_img)
    
    cropped_img2 = crop_image(flipped_img)
#
    imgs = cropped_img + cropped_img2
#    imgs = cropped_img
    imgs2 = []
    for i in imgs:
        i = resize_image(i)
        imgs2.append(i)
    
#        print(i.shape)
#        cv2.imshow('crop{}'.format(i), i)
#    print(len(imgs))
#    cv2.waitKey()
#    print(type(imgs2))
    imgs2 = np.array(imgs2) #change list to numpy array
#    print(type(imgs2))

    return imgs2

if __name__ == '__main__':
        img = imageio.imread('Presentation/image_dog.png') #image read in BGR
        data_aug(img)
