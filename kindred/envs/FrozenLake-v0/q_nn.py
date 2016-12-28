import numpy as np

import gym

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam, SGD

from kindred.util.numputil import Numputil

env = gym.make('FrozenLake-v0')

##########################

# Model Hyperparameters
MODEL = Sequential()

MODEL.add(Dense(env.observation_space.n, input_dim=1))
MODEL.add(PReLU())

MODEL.add(Dense(env.action_space.n))
MODEL.add(Activation('linear'))

MODEL.compile(Nadam(), 'mse')

MODEL.summary()

# Environment
EPISODES = 2000
FRAMES = 100

# Reward
GAMMA = .99

# Exploration
EPSILON_MAX = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = .0015

##########################

epsilon = EPSILON_MAX
episode_rewards = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for frame in range(FRAMES):
        #env.render()

        Q = MODEL.predict(Numputil.lT([state])).flatten()
        action = np.argmax(Q) if np.random.rand() > epsilon else env.action_space.sample()

        n_state, reward, done, info = env.step(action)
        n_Q = MODEL.predict(Numputil.lT([n_state])).flatten()
        Q[action] = reward + GAMMA*np.max(n_Q)

        MODEL.train_on_batch(Numputil.lT([state]), Q[np.newaxis])

        total_reward += reward

        if done:
            # print("Episode finished after {} timesteps".format(frame+1))
            break

        state = n_state

    epsilon = max(EPSILON_MIN, epsilon-EPSILON_DECAY)

    episode_rewards.append(total_reward)

    # print(episode_rewards)
    # print(Q)
    # print("Epsilon:", epsilon)

    if len(episode_rewards) < 100:
        continue

    avg = sum(episode_rewards[-100:])/float(100)
    print("Average reward over last 100 episodes: {}, episodes: {}".format(avg, episode))

    if avg > 0.77:
        print("Environment solved!")
        break
