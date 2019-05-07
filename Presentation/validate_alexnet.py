import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from win32api import GetSystemMetrics
import classify as cls

import cv2
#from alexnet import alexnet
classList=['bullfrog', 'Rana catesbeiana','Egyptian cat',
'basketball',
'plunger',' plumbers helper',
'reel'
'rocking chair, rocker',
'volleyball',
'lemon',
'espresso'
'cliff, drop, drop-off']


images = []
dogimg = cv2.imread('C:\\Users\\Admin\\Documents\\GitHub\\CNN-TinyImagenet-new\\Presentation\\image_dog.png')##what's type of image in imagent
inputimg = cv2.imread('C:\\Users\\Admin\\Documents\\GitHub\\CNN-TinyImagenet-new\\Presentation\\image_dog.png',0)##what's type of image in imagent

screen_res =  GetSystemMetrics(0),  GetSystemMetrics(1)
scale_width = screen_res[0] / dogimg.shape[1]
scale_height = screen_res[1] / dogimg.shape[0]
scale = min(scale_width, scale_height)
window_width = int(dogimg.shape[1] * scale)
window_height = int(dogimg.shape[0] * scale)

#uncomment below lines to show the image
cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
cv2.resizeWindow('dst_rt', window_width, window_height)


cv2.imshow('dst_rt', dogimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

inputimg=cv2.resize(inputimg,(28,28),interpolation = cv2.INTER_AREA)
inputimg = np.expand_dims(inputimg, axis=0) 

prob = cls.classify(inputimg,[],[], 1, 2, 2)
#
## birdimg = cv2.imread('image_bird.png')
## appleimg = cv2.imread('image_apple.png')
#images.append(dogimg)
## images.append(birdimg)
## images.append(appleimg)
#
#predictions = [];#use to store probabilities for each image
#
### apply to pretrained model
##for i in enumerate(images):
##    model = alexnet(images)
##    predictions.append(model)
#predictions.append( [{'dog':0.9},{'bird':0.12},{'apple':0.05}])
## predictions.append( [{'bird':0.87},{'apple':0.33},{'bird':0.1}])
## predictions.append( [{'apple':0.88},{'bird':0.19},{'dog':0.15}])
#
#
### plot name of class and confidence
## dogimg = cv2.resize(dogimg, (1800,2000))
#plt.subplot(221)
#plt.imshow(cv2.cvtColor(dogimg, cv2.COLOR_BGR2RGB))
#plt.title('Dog')
## plt.text(0, 2000, str(predictions[0]), size=10, rotation=0.,ha="left",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
#plt.axis('off')
##
## plt.show()
#
#
##Bar Graph
label = classList
prob= prob.tolist()


prob=[item[0] for item in prob]
plt.barh(label, prob)
plt.xlabel('Probability', fontsize=5)
plt.ylabel('Class', fontsize=5)
plt.xticks(label, prob, fontsize=2, rotation=30)
plt.title('Classification Result')
plt.show()






