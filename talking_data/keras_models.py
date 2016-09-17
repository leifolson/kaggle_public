model = Sequential()
model.add(Dense(output_dim=101, input_dim=21527, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=50, input_dim=101, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(tdk.Xtrain.toarray(), np_utils.to_categorical(tdk.y), nb_epoch=10)

# train acc: 0.1943
# test acc: 0.1873
model = Sequential()
model.add(Dense(output_dim=100, input_dim=21527, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(output_dim=50, input_dim=100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(Xtrain.toarray(), np_utils.to_categorical(yTrain), nb_epoch=10)

# train acc: 0.2699 
# test acc: 0.1955
model = Sequential()
model.add(Dense(output_dim=100, input_dim=21527, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(output_dim=50, input_dim=100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(Xtrain.toarray(), np_utils.to_categorical(yTrain), nb_epoch=10)


# train acc: 0.2470 
# test acc: 0.2014
model = Sequential()
model.add(Dense(output_dim=100, input_dim=21527, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=50, input_dim=100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(Xtrain.toarray(), np_utils.to_categorical(yTrain), nb_epoch=10)


# train acc: 0.2262 
# test acc: 0.2031
model = Sequential()
model.add(Dense(output_dim=100, input_dim=21527, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=50, input_dim=100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(Xtrain.toarray(), np_utils.to_categorical(yTrain), nb_epoch=10)


# train acc: 0.2117
# test acc: 0.1956
model = Sequential()
model.add(Dense(output_dim=100, input_dim=21527, activation='relu'))
model.add(Dense(output_dim=50, input_dim=100, activation='relu'))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(Xtrain.toarray(), np_utils.to_categorical(yTrain), nb_epoch=10)


'''
After some pca using truncated svd
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=200) ~75%
'''
# train acc: 0.1955 
# test acc: 0.1863
model = Sequential()
model.add(Dense(output_dim=100, input_dim=200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=50, input_dim=100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(svd.transform(Xtrain), np_utils.to_categorical(yTrain), nb_epoch=10)


# train acc: 0.2022
# test acc: 0.1842
model = Sequential()
model.add(Dense(output_dim=100, input_dim=200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=50, input_dim=100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=50, input_dim=50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(svd.transform(Xtrain), np_utils.to_categorical(yTrain), nb_epoch=20)

# went to 500 components
# train acc: 0.2231
# test acc: 0.1841
model = Sequential()
model.add(Dense(output_dim=100, input_dim=500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=50, input_dim=100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=50, input_dim=50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(svd.transform(Xtrain), np_utils.to_categorical(yTrain), nb_epoch=20)


# train acc: 0.3752 high variance (i.e., overfitting)
# test acc: 0.1772
model = Sequential()
model.add(Dense(output_dim=1000, input_dim=500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=500, input_dim=1000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=50, input_dim=500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(svd.transform(Xtrain), np_utils.to_categorical(yTrain), nb_epoch=20)


# train acc:  
# test acc: 
model = Sequential()
model.add(Dense(output_dim=1000, input_dim=500, activation='relu',  W_regularizer=l2(0.002)))
model.add(Dropout(0.3))
model.add(Dense(output_dim=500, input_dim=1000, activation='relu',  W_regularizer=l2(0.002)))
model.add(Dropout(0.3))
model.add(Dense(output_dim=50, input_dim=500, activation='relu',  W_regularizer=l2(0.002)))
model.add(Dropout(0.3))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid',  W_regularizer=l2(0.002)))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(svd.transform(Xtrain), np_utils.to_categorical(yTrain), nb_epoch=10)


model = Sequential()
model.add(Dense(output_dim=100, input_dim=500, activation='relu',  W_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(output_dim=50, input_dim=100, activation='relu',  W_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(output_dim=50, input_dim=50, activation='relu',  W_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid',  W_regularizer=l2(0.001)))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(svd.transform(Xtrain), np_utils.to_categorical(yTrain), nb_epoch=20)




####

model = Sequential()
model.add(Dense(output_dim=101, input_dim=21527, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(output_dim=50, input_dim=101, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(output_dim=50, input_dim=50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(tdk.Xtrain.toarray(), np_utils.to_categorical(tdk.y), nb_epoch=20)



model = Sequential()
model.add(Dense(output_dim=151, input_dim=21527, activation='relu', W_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(output_dim=101, input_dim=151, activation='relu', W_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(output_dim=50, input_dim=101, activation='relu', W_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(output_dim=50, input_dim=50, activation='relu', W_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid', W_regularizer=l2(0.01)))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(tdk.Xtrain.toarray(), np_utils.to_categorical(tdk.y), nb_epoch=40)


model = Sequential()
model.add(Dense(output_dim=151, input_dim=21527, activation='relu', W_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(output_dim=101, input_dim=151, activation='relu', W_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(output_dim=50, input_dim=101, activation='relu', W_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(output_dim=50, input_dim=50, activation='relu', W_regularizer=l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(output_dim=12, input_dim=50, activation='sigmoid', W_regularizer=l2(0.01)))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit(tdk.Xtrain.toarray(), np_utils.to_categorical(tdk.y), nb_epoch=40)
