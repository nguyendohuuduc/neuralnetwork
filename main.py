#Duc Nguyen
#main.py: main program of the experiments

import numpy as np
import prep as prep
import neuralnet as nn

#experiment 1
np.random.seed(0)
traindata = "838.arff"
testdata = "838.arff"
learningRate = 0.1
iter = 3000
x = range(iter)
#initialize and train the network
model = nn.neuralnet(3,1,traindata,testdata)
errtrain, errtest = model.train(iter,learningRate)
#print the hidden representation and plot training error rate
model.printHiddenRepresentation(1)
prep.graph(errtrain,x,"plot trainerror vs iterations on "+traindata+" 8-3-8 neural net ","training error rate","number of iterations","838.png")

#experiment 2
traindata = "optdigits_train.arff"
testdata = "optdigits_test.arff"
iter = 200
d = [3,3,3,3,3,3,0,1,2,3,4,5]
w = [5,10,15,20,30,40,10,10,10,10,10,10]
x = range(iter)
errtestList = []
#initialize and train the network with different depths and widths and plot the training and testing error vs iterations
for i in range(len(d)):
    model = nn.neuralnet(w[i],d[i],traindata,testdata)
    errtrain, errtest = model.train(iter,learningRate)
    errtestList.append(errtest[-1])
    prep.graph2(errtrain,errtest,x,"error vs iterations on "+traindata+" neuralnet "+str(w[i])+" width & "+str(d[i])+" depth","train error","test error","error rate","number of iterations","err_vs_iters_"+traindata+"_network_"+str(d[i])+"_"+str(w[i])+".png")

#plot test error vs depth and width of the neural network
err1 = errtestList[:len(errtestList)/2]
err2 = errtestList[len(errtestList)/2:]
_w = w[:len(w)/2]
_d = d[len(d)/2:]    
prep.graph(err1,_w,"plot of testing error vs width per layer of neural network on "+traindata,"testing error rate","width per layer of neural network","testerr_vs_width_"+traindata+".png")
prep.graph(err2,_d,"plot of testing error vs depth of neural network on "+traindata,"testing error rate","depth of neural network","testerr_vs_depth_"+traindata+".png")



