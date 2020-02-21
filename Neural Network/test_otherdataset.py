# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:50:31 2019

@author: wwong
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from studNet import NeuralNetwork
from studNet import sigderiv

#import instrNet
#import studentNN
np.random.seed(0)
data = datasets.make_circles(n_samples=100, noise=0.1)
lr = 0.1


X = data[0]
y = np.expand_dims(data[1], 1)
print(X)
print(y)


