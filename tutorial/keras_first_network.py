from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)


# load pima indians dataset
dataset = numpy.loadtxt("tutorial/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:-10,0:8]
Y = dataset[:-10,8]


# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8*10, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X, Y, epochs=10000, batch_size=1000)

print("evaluate the model")


nX = dataset[-10:,0:8]
nY = dataset[-10:,8]

# evaluate the model
scores = model.evaluate(nX[:,0:8], nY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# print("\n%s: %.2f%%" % (model.metrics_names[1], model.evaluate(nX, nY)[1]*100))
