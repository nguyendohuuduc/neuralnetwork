#Duc Nguyen 
#interface.py: interface to use the neural network

import numpy as np
import prep as prep
import neuralnet as nn
import matplotlib.pyplot as plt

np.random.seed(0)
learningRate = 0.1

w = int(input("Please specify the width of each layer of the network\n"))
d = int(input("Please specify the depth of the hidden layers of the network\n"))
train = str(input("Please specify the name of the training file you wish to train on.Must include double quotes\n"))
test = str(input("Please specify the name of the testing file you wish to test on.Must include double quotes\n"))
iteration = int(input("Please specify the number of iterations you wish to train on\n"))
model = nn.neuralnet(w,d,train,test)
errtrain,errtest = model.train(iteration,learningRate)
x = range(iteration)
plt.plot(x,errtrain,label="training error")
plt.plot(x,errtest,label="testing error")
plt.xlabel("number of iterations")
plt.ylabel("error rate")
plt.legend(loc=7)
plt.show()
    
