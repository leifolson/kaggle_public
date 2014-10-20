'''
    Author: Clinton Olson
    Email: clint.olson2@gmail.com

    The DataScaler class provides a number of data scaling techniques
    for array-like objects where rows are samples and columns are 
    variables
'''
import pandas as pd

class DataScaler:

    # init object
    def __init__(self,data):
        self.data = pd.DataFrame(data)

    # range scaling
    def rangeScale(self,cols):
        # get min/max values
        minVals = self.data.ix[:,cols].min()
        maxVals = self.data.ix[:,cols].max()

        # compute range
        rangeVals = maxVals - minVals

        # scale the features
        return ((self.data.ix[:,cols] - minVals) / rangeVals)

        

    # zero mean, unit variance scaling


    # get original data
    def getData(self):
        return self.data

    # set data
    def setData(self,data):
        self.data = pd.DataFrame(data)
