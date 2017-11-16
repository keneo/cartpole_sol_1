# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import numpy as np
# fix random seed for reproducibility
#numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("tutorial/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:-100,0:8]
Y = dataset[:-100,8]
tX = dataset[-100:,0:8]
tY = dataset[-100:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=1500, batch_size=1000)
# evaluate the model
scores = model.evaluate(tX, tY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.predict(X[:3])
Y[:3]

from sklearn.svm import SVR
import matplotlib.pyplot as plt
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

svr_lin = SVR(kernel='linear')
svr_poly = SVR(kernel='poly', degree =2)
svr_rbf = SVR(kernel='rbf', gamma=0.1)

svr_lin.fit(X, Y)
svr_poly.fit(X, Y)
svr_rbf.fit(X, Y)

def acc(model):
    return 1-np.sum(np.abs(np.abs(np.round(model.predict(tX)))-tY))/len(tX)

acc(svr_lin)
acc(svr_poly)
acc(svr_rbf)
acc(model)

svr_poly.predict(X[:10])
svr_rbf.predict(X[:10])
