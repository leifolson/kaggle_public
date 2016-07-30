import numpy as np
from sklearn import tree
from sklearn.decomposition.pca import PCA
import mnist_loader as loader
import mnist_writer as writer

print('Reading data...')
train_data, train_labels = loader.load_train_data()
test_data = loader.load_test_data()

# convert to numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)

print('PCA analysis...')
pca = PCA(n_components=35, whiten=True)
pca.fit(train_data)
train_data = pca.transform(train_data)
test_data = pca.transform(test_data)

print('Fitting decision tree...')
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_labels)

print('Making predictions...')
predict = clf.predict(test_data)

print('Writing results...')
writer.write_predictions(predict, '/Users/clint/Development/data/mnist/predict_tree.csv')