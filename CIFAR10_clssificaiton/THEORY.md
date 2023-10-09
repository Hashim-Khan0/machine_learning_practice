this case study projet will explore convolutional neural networks to classify images using the CIFAR-10 dataset. 
It consists of images divided into categories of different vehilces, animals etc.

## Learning Objectives
**The purpose of this project is to**
- understand the intuition behind deep convolutional nerual networks
- understand the basics of digital image processing
- perfrom 2D convolutions
- max pooling and average pooling
- build a CNN model to perform image classification using Keraas
- optimize network weights using Adam Optimzier
- perform network regularization techniques such as dropouts
- evaluate the model and present results using confusion matrices and classification reports
- perform image agumentation to enhance network generalization capability

## CIFAR10 classification
- Canadian Institute For Advanaced Research
- contans 60 0000 32x32 color images
- 6 000 images of each class(10 classes)
- images have low resolution
- we will create a classifier that takes the image as input and it will output a class
- dog.png ---> "Dog"

## What are CNNs and How Do They Learn
- ANN: it mimmicks the human brains nuerons
- the input is multiplied by their respective weights and add the bias signal then apply an activation function to that sum
- we cannot feed images to the ANN so we need to add a convolutional layer before we feed
- we start with the image and feed it into the convolutional layer and recieve different variations of the image
- then we apply a relu function(rectified linear unit) and feed it into the pooling layer(compressing an image to reduce the size of the feature map 32x32 --> 16x16)
- after pooling we will flatten the pixels(into 1 array)

### Convolutional Layer: Feature Detection
- convolutionals use a kernal matrix to scan a given image and apply a filter to obtain a certain effect
- an image kernal is a matrix used to apply effects such as blurring and sharpening
- kernels used in ML for feature extraction to select most important pxiels of an image
- convlution perserves the spatial relationship between pixels
- feature detectors are a matrix of size nxn where the values are multipled by the pixels of the image
- the sum is then stored in the feature map and this is repeated till all pixels have been covered of the large vector

### RELU (rectified linear units)
- RELU layers are used to add non-linearity to the feature map
- it also enhaces the sparsity or how scattered the feature map is
- the gradient of the RELU does not vanish as we increase x unlike the sigmoid function
- our feature map after applying RELU will have no negative values as they will be set to 0

### Pooling(downsampling)
- pooling or downsampling layers are placed after convolutional layers to reduce feature map dimensionality
- this improves the computational efficency while preserving the features
- pooling helps the model to generalize by avoiding overfitting
- if one pixel is shifted, the pooled feature map will be the same
- max pooling works by retaining the max feature response withing a give sample size in a feautre map
- STRIDE = 2 : take the max value in the 2x2 slice of matrix nxn
- the matrix is flattened and fed into the network

## How to Improve Perfromance of Network

### Increase Filters/Dropout
- improve accuracy by adding more feature detectors/filters or adding a dropout
- dropout refers to droppoing out units(neurons) in a neural network
- when training a network neurons develop co-dependency amongst each other during training
- dropout is a regularization technique for reducing overfitting in neural networks
- it enables training to occur on severl architectures of the nerual network

### Confusion Matrix
- rows are predictions 
- columns are true values
- we are looking for true positves and true negatives Q(1,1) & P(2,2)
- type 1 errors are false positve
- type 2 erros are false negative (WORSE)

### Key performance indicators
- classication accurace = (TP+TN) / (TP+TN+FP+FN)
- misclassifcation rate(error rate) = (FP+FN) / (TP+TN+FP+FN)
- percision = TP / (TP+FP) ; when the model predicted TRUE, how often was it right?
- Recall = TP / (TP+FN) ; when the class was actually TRUE how often did the classifier get it right?