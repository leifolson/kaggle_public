'''
    Author: Clinton Olson
    Email: clint.olson2@gmail.com

    The NeuralNet class implements a neural network model.
    The network can have an arbitrary number of layers of a specified size.  
    An overriding goal is to provide a general implementation

    Data, X, is assumed to have examples in columns.
    i.e. if we have N training examples, each of dimension M, then
    X will be MxN
'''

import numpy as np

class NeuralNet:

    '''
    Initialize network
    layers => vector of layer dimens
    '''          
    def __init__(self,layers, epochs):
        # get inputs
        self.layers = layers
        self.epochs = epochs

        # initialize small random network weights
        self.weights = self.initWeights(layers)


    '''
    Train Network
    '''
    def trainNet(self,data,eta):

        return

    '''
    Network Predict
    '''
    def netPredict(self,data):

        return

    '''
    Forward propagation
    '''

    '''
    Back propagation
    '''

    '''
    Set epochs
    '''
    def setEpochs(self,epochs):
        self.epochs = epochs
    
    '''
    Set layers
    '''
    def setLayers(self,layers):
        self.layers = layers
        self.weights = self.initWeights(layers)

    '''
    Init weights

        Weights are initialized to small random pos/neg numbers.
        The weights for each layer are stored in a list where 
        list element corresponds to the layer of the same index

        Weights are stored as column vectors for each hidden node       
    '''
    def initWeights(self,layers):

        # for each layer, create the corresponding weights matrix
        # and store into the list
        weights = []
        
        for i in range(len(layers)-1):
            rowDim = layers[i]
            colDim = layers[i+1]
            weights.append(np.random.rand(rowDim+1,colDim)*0.1-0.05)

        return weights

