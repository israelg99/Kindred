import os

import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

NAME = os.path.splitext(__file__)[0]

env = gym.make('CartPole-v0')

# Keras Model
model = Sequential()

model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(PReLU())

model.add(Dense(16))
model.add(PReLU())

model.add(Dense(16))
model.add(PReLU())

model.add(Dense(env.action_space.n))
model.add(Activation('linear'))

model.summary()

# Agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=32,
               target_model_update=1e-2, policy=policy)
dqn.compile(Nadam(), metrics=['mae'])
# dqn.load_weights(NAME + '_trained.h5')

# Training
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
dqn.save_weights(NAME + '.h5', overwrite=True)
dqn.test(env, nb_episodes=5, visualize=True)
