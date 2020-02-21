# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 01:05:01 2019

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
data = datasets.make_moons(n_samples=100, noise=0.1)
lr = 0.1


X = data[0]
y = np.expand_dims(data[1], 1)



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, )

xPass = x_train.T
yPass = y_train.T
#studentNet = instrNet.test_train(xPass, yPass)
#print(xPass)
#print(yPass)

inputSize = np.size(xPass, 0)
retNN = NeuralNetwork([inputSize, 4, 1])
[a, pre_activations, activations] = retNN.forward(xPass)

#print(pre_activations)
#print(activations)
#print(retNN.calcDeltas(activations,yPass))

cost_aout = activations[2] - yPass
aout_zout = sigderiv(pre_activations[1])
zout_wout = activations[1]
delta = (cost_aout*aout_zout).T
#print(delta)
for a in delta:
    retNN.biases[1] -= lr * a
#print(aout_zout)
#print(activations[2])
#print(activations[2].size)
#test2 = np.divide((activations[2] - y).T,((np.ones(activations[2].size)-activations[2])*activations[2]).T)
#testre= (activations[2] - y)/((np.ones(activations[2].size)-activations[2])*activations[2])
#print((activations[2]-yPass)/((np.ones(activations[2].size)-activations[2])*activations[2]))
#print(retNN.weights)

print((activations[2]-yPass)/activations[2])
cost_wout = np.dot(zout_wout, (cost_aout * aout_zout).T)
#print(cost_wout)


zh_wh = xPass
ah_zh =  sigderiv(pre_activations[0])
cost_ah = np.dot((aout_zout*cost_aout).T, retNN.weights[1]  )
cost_wh1 = np.dot(zh_wh,ah_zh.T  *cost_ah   )
#print(cost_wh1)
#print('new wh' , retNN.weights[0] - lr * cost_wh1.T)
#print('new wo' ,retNN.weights[1] - lr * cost_wout.T)
retNN.weights[0] -= lr * cost_wh1.T
retNN.weights[1] -= lr * cost_wout.T
#print(cost_wh1)
#print(retNN.biases[0])
delta =  cost_ah*ah_zh.T   
#print(delta)
for b in delta:
    retNN.biases[0] -= lr * a
    
    
    













