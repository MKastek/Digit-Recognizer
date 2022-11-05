# Digit-Recognizer
My solution for Kaggle competition: https://www.kaggle.com/competitions/digit-recognizer  

![alt text](https://github.com/MKastek/Digit-Recognizer/blob/97dffedc9eaf4c1c7d3265c6f5f73d656ab05e4b/images/mnist_dataset.png)

Based on [An Ensemble of Simple Convolutional Neural Network Models for MNIST Digit Recognition](https://arxiv.org/abs/2008.10400).  

The model was built based on convolutional neural networks:

- CNN: in_channels 1 kernel 3x3, stride 1x1, padding 1x1, out_channles 16,
- Batch Normalization 16 channels,
- CNN: in_channels 16 kernel 3x3, stride 1x1, padding 1x1, out_channles 32,
- Batch Normalization 32 channels,
- Linear Fully Connected Layer.
