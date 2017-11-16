import gym
import time
import itertools
import math

from keras.models import Sequential, model_from_json

from keras.layers import Dense


import numpy as np
# fix random seed for reproducibility
# np.random.seed(7)



env = gym.make('CartPole-v0')
# env.seed(0)

# print("action space: {}".format(env.action_space))


import pickle

episode_lessons = pickle.load(open( "episodes_lessons.p", "rb" ))


#
# # create model
# model = Sequential()
# # model.add(Dense(12, input_dim=5, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dense(4, activation='sigmoid'))
#
# # model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='linear'))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dense(1, activation='linear'))
#
# model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='linear'))
# model.add(Dense(8, activation='softmax'))
# model.add(Dense(1, activation='linear'))
#
# # Compile model
# # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

model = loaded_model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


all_episodes_dataset = None

index_to_minimize = 3
def function_to_optimize(ob):
    return [math.fabs(ob[3])]
           #+math.fabs(ob[0])

num_episodes = 100

for i in range(num_episodes):
    ob = env.reset()
    env.render()
    prev_state = ob
    for frame_no in range(1000):
        # action = agent.act(ob, reward, done)

        if False:  # i == 0:
            action = env.action_space.sample()
            ob_predicted = [0,0,0,0]
        else:
            potential_actions = list(range(0, 2))
            # potential_actions = list(map(lambda x: x / 10.0, range(0, 11)))
            test_lines = np.array(list(map(lambda a: list(itertools.chain(prev_state, [a])), potential_actions)))
            predicted = model.predict_on_batch(test_lines)

            #predicted_plus_fitness = np.concatenate((predicted.T, map(function_to_optimize, predicted)), axis=0)
            # bestindex = np.abs(predicted_plus_fitness).argmin(0)[index_to_minimize]
            #bestindex = function_to_optimize(np.abs(predicted - 0)).argmin(0)

            bestindex = np.abs(predicted).argmin(0)[index_to_minimize]

            nextaction = bestindex  # * 0.1

            action = nextaction
            ob_predicted = predicted[bestindex]
        # action = 0
        print("lets do: {}".format(action))
        prev_state = ob
        ob, reward, done, _ = env.step(action)
        # print("ob, reward, done: {}".format([ob, reward, done]))
        # print("ob predicted, actual, error: {}".format((ob_predicted[index_to_minimize], ob[index_to_minimize], (ob_predicted[index_to_minimize]-ob[index_to_minimize]))))
        ob_predicted_value = function_to_optimize(ob_predicted)[0]
        ob_value = function_to_optimize(ob)
        #print("ob predicted, actual, error: {}".format((ob_predicted_value, ob_value, (ob_predicted_value - ob_value))))

        env.render()

        result_state = ob

        learn_line = [prev_state, [action], result_state]
        episode_lessons += [learn_line]

        if done:
            print("episode finished at frame: %s" % frame_no)
            break

import pickle

pickle.dump(episode_lessons,open( "episodes_lessons2.p", "wb" ))
