import cv2

'''
    1. 227 or 224?
    2. channel first or last?
'''
IMAGE_SIZE = 256
CROP_SIZE = 224
'''
    @input: any image with size equal to or larger than 256,
    @return: a list of 2^4 images of the size 224x224
'''
#resize the image to 256x256
def resize_image(img):
    resized_img = cv2.resize(img, (256, 256))
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
#        cv2.imshow('crop{}'.format(i), i)
#    cv2.waitKey()
    return imgs


def data_aug(imgPath):
    img = cv2.imread(imgPath) #image read in BGR
#    cv2.imshow('original image', img)

    resized_img = resize_image(img)
#    cv2.imshow('resized image', resized_img)

    flipped_img = flip_image(resized_img)
#    cv2.imshow('flipped image', flipped_img)


    cropped_img = crop_image(resized_img)
    cropped_img2 = crop_image(flipped_img)

    imgs = cropped_img + cropped_img2
    print len(imgs)
    return imgs


data_aug('Presentation/image_dog.png')
