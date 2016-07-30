import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import mnist_loader as loader
import mnist_writer as writer

COMPONENT_NUM = 35

print('Read training data...')
train_data, train_label = loader.load_train_data()

print('Loaded ' + str(len(train_label)))

print('Reduction...')
train_label = numpy.array(train_label)
train_data = numpy.array(train_data)
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train_data)
train_data = pca.transform(train_data)

print('Train SVM...')
svc = SVC()
svc.fit(train_data, train_label)

print('Read testing data...')
test_data = loader.load_test_data()

print('Loaded ' + str(len(test_data)))

print('Predicting...')
test_data = numpy.array(test_data)
test_data = pca.transform(test_data)
predict = svc.predict(test_data)

print('Saving...')
writer.write_predictions(predict, '/Users/clint/Development/data/mnist/predict.csv')

