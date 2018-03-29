#Duc Nguyen
#prep.py: preprocess data and graph the result


import numpy as np
import matplotlib.pyplot as plt

#take a data file, and return a matrix consisting of data and all the labels
def readDataIntoMatrix(fileName):
    f = open(fileName, "r")
    numFeatures = 0
    numHeaderRows = 0
    numClasses = 0
    classes = 0
    attributeList = ["@ATTRIBUTE","@attribute"]
    dataList = ["@DATA","@data"]
    #read in number of header rows, attributes and number of labels
    for line in f:
        numHeaderRows = numHeaderRows + 1
        if all(sequence not in line for sequence in dataList):   
            if any(sequence in line for sequence in attributeList):
                numFeatures = numFeatures + 1
            if "class" in line:
                startchar = "{"
                endchar = "}"
                classes = (line[line.find(startchar)+1 : line.find(endchar)]).split(",")
        else:        
            break
    f.close()
    data = np.genfromtxt(fileName, delimiter = ',', dtype = "str",skip_header = numHeaderRows).reshape(-1, numFeatures)
    data = data.astype(float)
    return data,classes

#plot a line on one graph
def graph(y,x,label,yaxis,xaxis,figname):
    plt.plot(x,y)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.suptitle(label)
    plt.savefig(figname)
    plt.close()

#plot 2 lines on one graph
def graph2(y1,y2,x,label,line1,line2,yaxis,xaxis,figname):
    plt.plot(x,y1,label=line1)
    plt.plot(x,y2,label=line2)
    plt.legend(loc=7)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.suptitle(label)
    plt.savefig(figname)
    plt.close()
    
