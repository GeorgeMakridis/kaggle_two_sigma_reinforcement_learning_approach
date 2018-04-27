import kagglegym
import numpy as np
from keras.initializers import normal

np.random.seed(1337)

import pandas as pd
import math
import matplotlib.pyplot as plt
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense,  merge
from keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import random

HIDDEN1_UNITS = 70
HIDDEN2_UNITS = 30


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='tanh')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='tanh')(h0)
        V = Dense(1, activation='tanh', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        model = Model(input=S, output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, model.trainable_weights, S


HIDDEN1_UNITS = 70
HIDDEN2_UNITS = 30


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = merge([h1, a1], mode='sum')
        h3 = Dense(HIDDEN2_UNITS, activation='tanh')(h2)
        V = Dense(action_dim, activation='tanh')(h3)
        model = Model(input=[S, A], output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S


class OU(object):
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)


def fill_nans(train, d_mean):
    groups = set(train['id'].values)
    train_df = pd.DataFrame()
    for group in groups:
        df = train.loc[train['id'] == group]
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        # df = sp.detrend(df[col], axis=-1, type='linear', bp=0)
        train_df = train_df.append(df)
    train_df.fillna(d_mean, inplace=True)
    return train_df


def get_reward(y_true, y_fit):
    R2 = 1 - np.sum((y_true - y_fit) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    R = np.sign(R2) * math.sqrt(abs(R2))
    return (R)


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


full_df = pd.read_hdf('../input/train.h5')

# The "environment" is our interface for code competitions
env = kagglegym.make()
o = env.reset()

excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col = [c for c in o.train.columns if c not in excl]
train = o.train[:]
d_mean = train.median(axis=0)
train = train.fillna(d_mean, inplace=True)

t = 0
history = dict()
r_add = 0

y_actual_list = []
y_pred_list = []
r1_overall_reward_list = []
ts_list = []

print ("--------------setup actor critic network---------------")

BUFFER_SIZE = 100000
BATCH_SIZE = 10000
GAMMA = 0.9
TAU = 0.00001  # Target Network HyperParameters
LRA = 0.0001  # Learning rate for Actor
LRC = 0.001  # Lerning rate for Critic
action_dim = 1  # Steering/Acceleration/Brake
state_dim = 108  # of sensors input
EXPLORE = 100000.
# episode_count = 2000
# max_steps = 100000
reward = 0
done = False
step = 0
epsilon = 1
indicator = 0
train_indicator = True
# Tensorflow GPU optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K

K.set_session(sess)

actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer
OU = OU()  # Ornstein-Uhlenbeck Process

actor.model.fit(train[col].values, train['y'].values, nb_epoch=5, batch_size=1000, verbose=2)
loss = critic.model.fit(
    [train[col].values, actor.model.predict(train[col].values)], train['y'].values, nb_epoch=5, batch_size=1000,
    verbose=2)
print(loss)
loss = 0
actor.target_train()
critic.target_train()


total_reward = 0
y_mean = train['y'].median(axis=0)
y_std = train['y'].std(axis=0)
y_mean += y_std / 2
print("y mean %s", y_mean)
print("y std %s", y_std)

while True:
    timestamp = o.features["timestamp"][0]
    actual_y = list(full_df[full_df["timestamp"] == timestamp]["y"].values)

    s_t = o.features[col]
    s_t = s_t.fillna(d_mean)
    loss = 0
    epsilon -= 1.0 / EXPLORE
    a_t = o.target
    noise_t = o.target

    a_t_original = actor.model.predict(s_t.values)
    a_t_original = a_t_original.flatten()
    noise_t['y'] = max(epsilon, 0) * OU.function(a_t_original, y_mean, 0.1, y_std)
    a_t['y'] = 0.9 * a_t_original + 0.1 * noise_t['y']
    o, reward, done, info = env.step(a_t)
    s_t1 = o.features[col]
    s_t1 = s_t.fillna(d_mean)
    buff.add(s_t, a_t, reward, s_t1, done)  # Add replay buffer

    # Do the batch update
    batch = buff.getBatch(BATCH_SIZE)
    states = pd.DataFrame()
    actions = pd.DataFrame()
    rewards = np.empty((0))
    new_states = pd.DataFrame()
    dones = np.empty((0))
    y_t = pd.DataFrame()
    i = 0
    for e in batch:
        states = states.append(e[0])
        actions = actions.append(e[1])
        rewards = np.append(rewards, e[2])
        new_states = new_states.append(e[3])
        dones = np.append(dones, e[4])
        y_t = y_t.append(e[1])
        i += 1

    target_q_values = critic.target_model.predict([new_states.values, actor.target_model.predict(new_states.values)])
    for k in range(len(batch)):

        if reward > 0:
            y_t.values[k] = target_q_values[k]
        else:
            if y_t.values[k].mean() > 0:
                y_t.values[k] = (1 - GAMMA) * rewards[k] + GAMMA * target_q_values[k]
            else:
                y_t.values[k] = -(1 - GAMMA) * rewards[k] + GAMMA * target_q_values[k]


    if (train_indicator):
        loss += critic.model.train_on_batch([states.values, actions['y'].values], y_t['y'])
        a_for_grad = actor.model.predict(states.values)
        grads = critic.gradients(states.values, a_for_grad)
        actor.train(states.values, grads)
        actor.target_train()
        critic.target_train()

    total_reward += reward
    s_t = s_t1
    step += 1

    overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))
    r1_overall_reward_list.append(overall_reward)
    ts_list.append(timestamp)

    if done:
        print(done)
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ts_list, r1_overall_reward_list, c='blue')
        plt.plot(ts_list, [0] * len(ts_list), c='red')
        plt.title("Cumulative R value change for Univariate Ridge (technical_20)")
        plt.ylim([-0.9, 0.04])
        plt.xlim([850, 1850])
        plt.show()
        print("el fin ...", info["public_score"])
        break
    if o.features.timestamp[0] % 1 == 0:
        print(reward)
