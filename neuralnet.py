#Duc Nguyen
#neuralnet.py: neural network class to train and test the model

import numpy as np
import prep as prep
from scipy.special import expit

class neuralnet:
    #initialize the model with width and depth of the hidden layers and train and test data
    def __init__(self, w, d, trainData, testData):
        #preprocess the data and read data into matrix
        traindata, classes = prep.readDataIntoMatrix(trainData)
        testdata, classes = prep.readDataIntoMatrix(testData)
        value = 0
        table = {}
        #transform labels into numbers from 0 to numClasses-1
        for i in range(len(classes)):
            classes[i] = int(classes[i])
        classes.sort()
        for cl in classes:
            table[cl]=value
            value = value + 1
        numClasses = len(classes)
        #set the member variable of the class
        self.w = w
        self.d = d
        self.Ntrain = traindata.shape[0]
        self.Ntest = testdata.shape[0]
        self.Nfeatures = traindata.shape[1]-1
        self.Xtrain = traindata[:,0:traindata.shape[1] - 1]
        self.ytrain = np.zeros((self.Ntrain,1))
        for i in range(self.Ntrain):
            self.ytrain[i,0] = table[int(traindata[i,self.Nfeatures])]
        self.Xtest = testdata[:,0:testdata.shape[1] - 1]
        self.ytest = np.zeros((self.Ntest,1))
        for i in range(self.Ntest):
            self.ytest[i,0] = table[int(testdata[i,self.Nfeatures])]
        self.nIn = self.Xtrain.shape[1]
        self.nOut = numClasses
        self.ytrainM = np.zeros((self.Ntrain, self.nOut))
        #transform labels into vector of zeros and ones for training
        for i in range(self.Ntrain):
            self.ytrainM[i,int(self.ytrain[i])] = 1
        #initialize weigts
        self.weights = []
        if self.d == 0:
            self.weights.append(np.random.uniform(-0.1, 0.1, (self.nIn, self.nOut)))
        else:
            self.weights.append(np.random.uniform(-0.1, 0.1, (self.nIn, self.w)))
            for i in range(d - 1):
                self.weights.append(np.random.uniform(-0.1, 0.1, (self.w, self.w)))
            self.weights.append(np.random.uniform(-0.1, 0.1, (self.w, self.nOut)))

    #train the network with iter iterations and learning rate learningRate, return a list of training and testing error in every iteration    
    def train(self, iter, learningRate):
        listTrainWrong = []
        listTestWrong = []
        for i in range(iter):
            numTrainWrong = 0
            for j in range(self.Ntrain):
                derivatives = []
                deltas = []
                xs = []
                s = 0
                x = self.Xtrain[j,:].reshape(1,-1)
                xs.append(x)
                #feedforward, taking in input and give out a prediction, storing derivatives ont the way 
                for k in range(self.d + 1):
                    s = x.dot(self.weights[k])
                    x = expit(s)
                    xs.append(x)
                    derivatives.append(x - np.multiply(x, x))
                dt = np.multiply(x - self.ytrainM[j,:], derivatives[self.d])
                #if prediction is wrong, add to the number of wrong training example
                predict = np.argmax(x)
                if predict != int(self.ytrain[j]):
                    numTrainWrong = numTrainWrong + 1
                deltas.append(dt)
                #back propagation, use later layer deltas to compute previous deltas and store them
                k = self.d
                while k > 0:
                    dt = np.multiply(derivatives[k - 1], dt.dot(np.transpose(self.weights[k])))
                    deltas.append(dt)
                    k = k - 1
                k = self.d
                #use the precomputed deltas to update the weights
                for p in range(len(deltas)):
                    self.weights[p] = self.weights[p] - learningRate * np.transpose(xs[p]).dot(deltas[k])
                    k = k - 1
            #test the current network on the test set and get the error rate
            numTestWrong = 0
            for i in range(self.Ntest):
                s = 0
                x = self.Xtest[i,:].reshape(1,-1)
                for k  in range(self.d + 1):
                    s = x.dot(self.weights[k])
                    x = expit(s)
                predict = np.argmax(x)
                if predict != int(self.ytest[i]):
                    numTestWrong = numTestWrong + 1
            listTrainWrong.append(float(numTrainWrong)/self.Ntrain)
            listTestWrong.append(float(numTestWrong)/self.Ntest)
        return listTrainWrong, listTestWrong

    #print out hidden representation tested on the testing set at specified layer
    def printHiddenRepresentation(self,layer):
        #for each example in test set, feed forward in the network and if it is at specified layer, print out the
        #representation at that layer
        for i in range(self.Ntest):
            l = 0
            s = 0
            x = self.Xtest[i,:].reshape(1,-1)
            for k in range(self.d + 1):
                s = x.dot(self.weights[k])
                x = expit(s)
                l = l + 1
                if l == layer:
                    print(x)
