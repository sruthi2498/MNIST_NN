# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import time
import sys

# from google.colab import drive
# drive.mount('/content/drive')

# !ls

# import os
# os.chdir("drive/My Drive/USC/AI")
# !ls

# sys.argv

# train_df_filename = "train_image.csv"
# train_labels_filename = "train_label.csv"
# test_df_filename = "test_image.csv"

train_df_filename = sys.argv[1]
train_labels_filename = sys.argv[2]
test_df_filename = sys.argv[3]

train_df = pd.read_csv(train_df_filename,header=None).values
train_labels = pd.read_csv(train_labels_filename,header=None).values
test_df = pd.read_csv(test_df_filename,header=None).values
# test_labels = pd.read_csv("test_label.csv",header=None).to_numpy()

size = train_df.shape[0]
features = train_df.shape[1]
test_size = test_df.shape[0]
# print(size,features, test_size)

def removeNan(X,y):
  indices = np.argwhere(np.isnan(X))

def normalizeData(X):
  return X/255

train_df = normalizeData(train_df).T
test_df = normalizeData(test_df).T
# train_df[:1]

labels = 10
def oneHotEncodeLabels(y,size):
  y = y.reshape(1, size)
  b = np.zeros((y.size, 10))
  b[np.arange(y.size),y] = 1
  return b.T

train_labels = oneHotEncodeLabels(train_labels,size)
# print(train_labels.shape)
# print(train_labels.shape)
# test_labels=oneHotEncodeLabels(test_labels,test_size)
# test_labels.shape

import matplotlib.pyplot as plt
import matplotlib

def plotRow(i):
  print("Label : ",train_labels[:,i])
  plt.imshow(train_df[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
  plt.show()
  
# plotRow(random.randint(1,size))
# print(train_df.to_numpy().T[:,2].shape)

def sigmoid(x):
    return 1./(1. + np.exp(-x))

# small_val = np.finfo(float).eps
small_val = 1e-9
large_val = np.finfo(float).max

def fix(x):
  x = x+0.
  x = np.where(x ==0, small_val, x) 
  x = np.where(x ==-math.inf, -large_val, x) 
  return np.where(x ==math.inf, large_val, x) 

def softmax(x):
  s = np.exp(x) / np.sum(np.exp(x), axis=0)
  return  fix(s)

# arr= np.array([-0.62968587, -0.62968587, -0.62968587, 0, -0., math.inf, -math.inf])
# arr,fix(arr)

def xavierInitializeWeights(prev_layer,size):
  l = -(math.sqrt(6.0) / math.sqrt(prev_layer + size))
  u = (math.sqrt(6.0) / math.sqrt(prev_layer + size))
  weights = np.random.rand(prev_layer)
  weights = l + weights * (u - l)
  return weights

def getWeightMatrix(prev_layer_size,size):
    weights = []
    for _ in range(size):
      weights.append(xavierInitializeWeights(prev_layer_size,size))
    weights = np.array(weights)
    return weights

def getCategoricalCrossEntropy(Y, Y_pred):
    L_sum = np.sum(np.multiply(Y, np.log(Y_pred)))
    m = Y.shape[1]
    #print(m)
    L = -(1/m) * L_sum
    return L

# Y = train_labels[:,:5]
# Z = np.random.randn(10,5)
# Y_pred = np.array([softmax(i) for i in Z])
# # Y_pred= softmax()
# train_labels.shape,Y.shape, Z.shape, Y_pred.shape, getCategoricalCrossEntropy(Y,Y_pred)

class Model: 
  def __init__(self, inputLayersSize, hiddenLayersSize, outputLayersSize):
    self.inputLayersSize = inputLayersSize
    self.hiddenLayersSize = hiddenLayersSize
    self.outputLayersSize = outputLayersSize
    self.loss = math.inf
    self.learning_rate = 0.5
    self.initializeWeights()
    self.initializeBiases()
    self.initializeOutputs()

  def initializeWeights(self):
    self.W1 = getWeightMatrix(self.inputLayersSize,self.hiddenLayersSize )
    self.W2 = getWeightMatrix(self.hiddenLayersSize,self.outputLayersSize )
    # print("W1 : ",self.W1.shape, "W2 : ",self.W2.shape)

  def initializeBiases(self):
    self.b1 = np.zeros((self.hiddenLayersSize,1))
    self.b2 = np.zeros((self.outputLayersSize,1))
    # print("b1 : ",self.b1.shape, "b2 : ",self.b2.shape)
  
  def initializeOutputs(self):
    self.Z1 = None
    self.Z2 = None 
    self.A1 = None 
    self.A2 = None

  def setInput(self,input,target):
    self.input = fix(input)
    self.target = target

  def performFeedForwardPass(self):
    #Z1 = W1X + b => sigmoid
    self.Z1 = fix(np.matmul(self.W1,self.input)  + self.b1)
    if np.isnan(np.sum(self.Z1)):
      print("self.Z1 has nan")
    self.A1 = fix(sigmoid(self.Z1))
    if np.isnan(np.sum(self.A1)):
      print("self.A1 has nan")
    #self.A1 = batchNormalize(self.A1)
    #print(self.A1.shape)
    self.Z2 = fix( np.matmul(self.W2,self.A1)  + self.b2)
    if np.isnan(np.sum(self.Z2)):
      print("self.Z2 has nan")
    # self.Z2 = np.log(np.max(self.Z2, 1e-9))
    self.A2 = fix(softmax(self.Z2))
    if np.isnan(np.sum(self.A2)):
      print("self.A2 has nan")
    #print(self.A2.shape)
    #print(self.A2,sum(self.A2))

  def calculateLoss(self):
    self.loss = getCategoricalCrossEntropy(self.target, self.A2)
    #print("Loss = ",self.loss)

  def getOutput(self):
    return self.output

  def performBackProp(self):
    m =  self.target.shape[1]
    diff_Z2 =  self.A2-self.target
    self.diff_W2 = fix((1./m) * np.matmul(diff_Z2, self.A1.T))
    self.diff_b2 = fix((1./m) * np.sum(diff_Z2, axis=1, keepdims=True))

    diff_A1 = np.matmul(self.W2.T, diff_Z2)

    diff_Z1 = diff_A1 * sigmoid(self.Z1)*(1 - sigmoid(self.Z1))
    self.diff_W1 = fix((1./m) * np.matmul(diff_Z1, self.input.T))
    self.diff_b1 = fix((1./m) * np.sum(diff_Z1,axis=1,keepdims=True))

  def updateWeightsAndBias(self):
    #print("Updating weights and bias")
    self.W1 = fix(self.W1 - self.learning_rate * self.diff_W1)
    if np.isnan(np.sum(self.W1)):
      print("self.W1 has nan")
    self.b1 = fix(self.b1 - self.learning_rate * self.diff_b1)
    if np.isnan(np.sum(self.b1)):
      print("self.b1 has nan")

    self.W2 = fix(self.W2 - self.learning_rate * self.diff_W2)
    if np.isnan(np.sum(self.W2)):
      print("self.W2 has nan")
    self.b2 = fix(self.b2 - self.learning_rate * self.diff_b2)
    if np.isnan(np.sum(self.b2)):
      print("self.b2 has nan")

  def getLoss(self):
    return self.loss

  def predictClass(self,input):
    #print(input.shape)
    Z1 = fix(np.matmul(self.W1, input)  + self.b1)
    A1 = fix(sigmoid(Z1))
    Z2 = fix( np.matmul(self.W2,A1)  + self.b2)
    A2 = fix(softmax(Z2))
    return np.argmax(A2, axis=0)

# n_hidden = 800
# n_output = 10
# model = Model(features,n_hidden,n_output)

def shuffle(a, b):
    #print(a.shape,len(a),b.shape,len(b))
  shuffler = np.random.permutation(len(a.T))
  return a.T[shuffler].T, b.T[shuffler].T

def getAccuracy(Y,Y_pred):
  accuracy = (Y == Y_pred).sum() / len(Y)
  #print("Accuracy ",accuracy)
  return accuracy

# batch_size = 256
# num_batchs = math.ceil(size//batch_size)
# size, num_batchs, num_batchs*batch_size

'''
X,y = shuffle(train_df,train_labels)
j = random.randint(0,size - batch_size)
input = X[:,j:j+batch_size]
target = y[:,j:j+batch_size]
#print(input.shape, target.shape)
model.setInput(input,target)
model.performFeedForwardPass()
model.calculateLoss()
model.performBackProp()
model.updateWeightsAndBias()
model.getLoss()'''

def trainOneEpoch(train_df,train_labels):
  t1 = time.time()
  batch_size = 32
  num_batchs = size//batch_size
  X,y = shuffle(train_df,train_labels)
  i = 0
  j =0
  #loss = []
  while i<num_batchs:
    #print("Batch",i+1)
    input = X[:,j:j+batch_size]
    target = y[:,j:j+batch_size]
    #print(input.shape, target.shape)
    model.setInput(input,target)
    model.performFeedForwardPass()
    model.calculateLoss()
    model.performBackProp()
    model.updateWeightsAndBias()
    #loss.append(model.getLoss())
    i+=1
    j+=batch_size
  if j<=size:
    input = X[:,j:]
    target = y[:,j:]
    #print(input.shape, target.shape)
    model.setInput(input,target)
    model.performFeedForwardPass()
    model.calculateLoss()
    model.performBackProp()
    model.updateWeightsAndBias()
  return time.time()-t1,model.getLoss()

# trainOneEpoch(train_df,train_labels)

n_hidden = 800
n_output = 10
model = Model(features,n_hidden,n_output)
totalTime = 0

loss = []
train_accuracy=[]
test_accuracy = []
epochs = 20
for i in range(epochs):
  t,l = trainOneEpoch(train_df,train_labels)
  totalTime+=t
  loss.append(l)

  train_df,train_labels = shuffle(train_df,train_labels)
  train_preds = model.predictClass(train_df)
  train_actual = np.argmax(train_labels, axis=0)
  train_acc = getAccuracy(train_actual,train_preds)
  train_accuracy.append(train_acc)
  # print("Epoch ",i, "loss = ",l, " train acc = ",train_acc)
  
  '''
  test_df,test_labels = shuffle(test_df,test_labels)
  test_preds = model.predictClass(test_df)
  test_actual = np.argmax(test_labels, axis=0)
  test_acc = getAccuracy(test_actual,test_preds)
  test_accuracy.append(test_acc)
  print("Epoch ",i, "loss = ",l, " train acc = ",train_acc, " test acc = ",test_acc)'''
  

# print("Total time = ",totalTime)

# plt.plot(loss)
# plt.show()

# plt.plot(train_accuracy)
# plt.plot(test_accuracy)

# plt.show()

test_preds = model.predictClass(test_df)
# test_preds

output_file = "test_predictions.csv"
np.savetxt(output_file, test_preds.astype(int), delimiter="\n")