# Digit-Recognizer
My solution for Kaggle competition: https://www.kaggle.com/competitions/digit-recognizer  

Based on [An Ensemble of Simple Convolutional Neural Network Models for MNIST Digit Recognition](https://arxiv.org/abs/2008.10400).  

The model was built based on convolutional neural networks:

- CNN: in_channels 1 kernel 3x3, stride 1x1, padding 1x1, out_channles 16,
- Batch Normalization 16 channels,
- CNN: in_channels 16 kernel 3x3, stride 1x1, padding 1x1, out_channles 32,
- Batch Normalization 32 channels,
- Linear Fully Connected Layer.
