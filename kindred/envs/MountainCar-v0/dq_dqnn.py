import os

import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from kindred.callbacks.epsilon_decay import EpsilonDecay

NAME = os.path.splitext(__file__)[0]

env = gym.make('MountainCar-v0')

# Keras Model
model = Sequential()

model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(4))
model.add(PReLU())

model.add(Dense(env.action_space.n))
model.add(Activation('linear'))

model.summary()

# Agent
memory = SequentialMemory(limit=100000, window_length=1)
policy = EpsGreedyQPolicy(1)
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, policy=policy,
               batch_size=64, nb_steps_warmup=128)
dqn.compile(Nadam(), metrics=['mae'])
# dqn.load_weights(NAME + '_trained.h5')

# Training
dqn.fit(env, nb_max_episode_steps=10000, nb_steps=1000000, visualize=False, verbose=2,
        callbacks=[EpsilonDecay(0.97, offset=0, skip=1, minimum=0.01)])

dqn.save_weights(NAME + '.h5', overwrite=True)
dqn.test(env, nb_episodes=5, visualize=True)
