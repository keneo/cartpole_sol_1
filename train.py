from keras.models import Sequential, model_from_json
from keras.layers import Dense
import numpy as np
import itertools
# fix random seed for reproducibility
#np.random.seed(7)



import pickle

episode_lessons = pickle.load(open( "episodes_lessons.p", "rb" ))

episode_dataset = np.array(list(map(lambda l: list(itertools.chain(l[0], l[1], l[2])), episode_lessons)))
all_episodes_dataset = episode_dataset
X = all_episodes_dataset[:, 0:5]
Y = all_episodes_dataset[:, 5:9]

with tf.device('/gpu:0'):

    model = Sequential()
    model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='linear'))
    model.add(Dense(80, activation='linear'))
    model.add(Dense(8, activation='linear'))
    model.add(Dense(4, activation='linear'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.load_weights("model.h5")

    #for i in range(100):
    for i in range(1):
        print("fit")
        model.fit(X, Y, epochs=1000, batch_size=100000, verbose=0)  #WTF is epoch?
        #model.fit(X, Y, epochs=10000, batch_size=100000, verbose=0)
        #model.fit(X[:1], Y[:1], epochs=1, batch_size=1, verbose=0)

        print("eval")
        scores = model.evaluate(X[:20], Y[:20])
        print (scores)
        print(model.metrics_names)
        #print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        pY = model.predict_on_batch(X)

        print(pY[:2])
        print(Y[:2])

        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

