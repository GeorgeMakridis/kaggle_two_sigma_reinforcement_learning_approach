import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import random
import numpy
from collections import deque
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense

def get_reward(y_true, y_fit):
    R2 = 1 - np.sum((y_true - y_fit) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    R = np.sign(R2) * math.sqrt(abs(R2))
    return (R)

full_df = pd.read_hdf('../input/train.h5')

CONFIG = 'nothreshold'
ACTIONS = 4  # number of valid actions
GAMMA = 0.9  # decay rate of past observations
OBSERVATION = 60  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50  # number of previous transitions to remember
BATCH = 15  # size of minibatch
FRAME_PER_ACTION = 1

global model

def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Dense(14, input_dim=7, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='mse', optimizer='rmsprop')
    print("We finish building the model")
    return model


env = kagglegym.make()
o = env.reset()
print(o.train.shape)
o.train = o.train[:]
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col = [c for c in o.train.columns if c not in excl]

col = ['technical_13', 'technical_10', 'technical_6', 'technical_27',
       'technical_14', 'technical_20', 'technical_30']

train = pd.read_hdf('../input/train.h5')

train = train[col]

d_mean = train.median(axis=0)

train = o.train[col]
train = train.fillna(d_mean)

rfr = ExtraTreesRegressor(n_estimators=50, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
model1 = rfr.fit(train, o.train['y'])
train = []

low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (o.train.y > high_y_cut)
y_is_below_cut = (o.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
model2 = LinearRegression(n_jobs=-1)
model3 = LinearRegression(n_jobs=-1)
model4 = LinearRegression(n_jobs=-1)
model2.fit(np.array(o.train[col].fillna(d_mean).loc[y_is_within_cut, 'technical_20'].values).reshape(-1, 1),
           o.train.loc[y_is_within_cut, 'y'])
model3.fit(np.array(o.train[col].fillna(d_mean).loc[y_is_within_cut, 'technical_30'].values).reshape(-1, 1),
           o.train.loc[y_is_within_cut, 'y'])
model4.fit(np.array(o.train[col].fillna(d_mean).loc[y_is_within_cut, 'technical_27'].values).reshape(-1, 1),
           o.train.loc[y_is_within_cut, 'y'])

ymean_dict = dict(o.train.groupby(["id"])["y"].median())

D = deque()
OBSERVE = OBSERVATION
t = 0
DROP = 700

model = buildmodel()

y_actual_list = []
y_pred_list = []
r1_overall_reward_list = []
ts_list = []

while True:
    timestamp = o.features["timestamp"][0]
    actual_y = list(full_df[full_df["timestamp"] == timestamp]["y"].values)
    loss = 0
    Q_sa = 0
    action_index = 0
    r_t = 0
    a_t = np.zeros([ACTIONS])

    test = o.features[col]
    test = test.fillna(d_mean)
    pred = o.target
    test2 = np.array(o.features[col].fillna(d_mean)['technical_20'].values).reshape(-1, 1)
    test3 = np.array(o.features[col].fillna(d_mean)['technical_30'].values).reshape(-1, 1)
    test4 = np.array(o.features[col].fillna(d_mean)['technical_27'].values).reshape(-1, 1)

    q = model.predict_proba(test.values)
    a_t = q.mean(axis=0)
    max_Q = np.argmax(abs(q), axis=1)
    action_index = np.round(max_Q.mean())


    if t > OBSERVE and t % 10 == 0:
        # sample a minibatch to train on
        minibatch = random.sample(D, BATCH)

        inputs = np.zeros((BATCH, DROP, test.shape[1]))  # 32, 80, 80, 4
        targets = np.zeros((inputs.shape[0], inputs.shape[1], ACTIONS))  # 32, 2

        # Now we do the experience replay
        for i in range(0, len(minibatch)):
            test_t = minibatch[i][0]
            pred_t = minibatch[i][1]  # This is action index
            reward_t = minibatch[i][2]
            test_t1 = minibatch[i][3]
            done = minibatch[i][4]

            test_t = test_t[:DROP]
            pred_t = pred_t[:DROP]
            test_t1 = test_t1[:DROP]

            inputs[i:i + 1] = test_t  # I saved down s_t

            targets[i] = model.predict_proba(test_t.values)  # Hitting each buttom probability
            test_t1 = test_t1.fillna(d_mean)
            Q_sa = model.predict_proba(test_t1.values)

            if done:
                targets[i, action_index] = reward_t
            else:
                targets[i] = reward_t / 4 + targets[i]
                targets[i, action_index] = (1 - GAMMA) * reward_t * 3 / 4 + GAMMA * Q_sa[action_index]

        inputs = numpy.reshape(inputs, (inputs.shape[0] * inputs.shape[1], inputs.shape[2]))
        targets = numpy.reshape(targets, (targets.shape[0] * targets.shape[1], targets.shape[2]))

        model.fit(inputs, targets, batch_size=30, nb_epoch=20, verbose=2)
    print(a_t)
    pred['y'] = model1.predict(test).clip(low_y_cut, high_y_cut) * (
    a_t[0] / (a_t[0] + a_t[1] + a_t[2] + a_t[3])) + model2.predict(test2).clip(low_y_cut, high_y_cut) * (
    a_t[1] / (a_t[0] + a_t[1] + a_t[2] + a_t[3])) + model3.predict(test3).clip(low_y_cut, high_y_cut) * (
    a_t[2] / (a_t[0] + a_t[1] + a_t[2] + a_t[3])) + model4.predict(test4).clip(low_y_cut, high_y_cut) * (
    a_t[3] / (a_t[0] + a_t[1] + a_t[2] + a_t[3]))

    o, reward, done, info = env.step(pred)
    pred_y = list(pred['y'].values)
    y_actual_list.extend(actual_y)
    y_pred_list.extend(pred_y)
    overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))
    r1_overall_reward_list.append(overall_reward)
    ts_list.append(timestamp)

    if done:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ts_list, r1_overall_reward_list, c='blue')
        plt.plot(ts_list, [0] * len(ts_list), c='red')
        plt.title("Cumulative R value change for Univariate Ridge (technical_20)")
        plt.ylim([-0.04, 0.04])
        plt.xlim([850, 1850])
        plt.show()
        print("el fin ...", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(t)
        print(reward)

    D.append((test, pred, reward, o.features[col], done))
    if len(D) > REPLAY_MEMORY:
        D.popleft()
    t = t + 1