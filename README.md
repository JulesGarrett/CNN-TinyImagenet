
# EECS-738-P3
## Convolutional Neural Network Based on AlexNet for Image Classification

### Team Members
- **Ian Yake**
- **Jules Garrett**
- **Jian Shen**
- **Dinesh Dandamudi**
- **Brian Quiroz**

### Overview
We created a CNN from Numpy based on AlexNet and used it to classify images from a dataset of 100,000 images into 200 categories corresponding to the object shown in the image. Our architecture and dataset are a scaled down version of AlexNet's as we didn't have the computing power to fully replicate the paper. After being trained on the [Tiny ImageNet dataset]([https://tiny-imagenet.herokuapp.com/](https://tiny-imagenet.herokuapp.com/)), our model outputs confidence levels for each category.

After replicating Alexnet, we experiment with the architecture and dataset to see what makes Alexnet work so well and how it can be improved.
Changes include:
* Number of layers
* Size of layers
* Re-adding images into training dataset with occlusion
* Re-adding images into the training dataset after they're passed through an edge detection filter

We seek to discover what patterns the model primarily exploits to classify images and how well such a model generalizes to different types of images (i.e., occluded or without texture).

### Approach


### Results


### Accuracy
*TBD*

### Requisites
The code was written from scratch using Python 3.7(?) and the following modules:

- Numpy
- cv2
- Pickle


### References
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
