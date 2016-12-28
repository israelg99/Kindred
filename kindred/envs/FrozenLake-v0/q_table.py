import numpy as np

import gym

##########################

# Environment
EPISODES = 2000
FRAMES = 100

# Learning
ALPHA_MAX = .95
ALPHA_MIN = .25
ALPHA_DECAY = .004

# Reward
GAMMA = .99

# Exploration
EPSILON_MAX = 1
EPSILON_MIN = 0
EPSILON_DECAY = .0015

##########################

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = ALPHA_MAX
epsilon = EPSILON_MAX
episode_rewards = []

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    for frame in range(FRAMES):
        #env.render()

        # Dynamic noise exploration.
        # action = np.argmax(Q[state] + np.random.randn(1, env.action_space.n)*(1.0/(episode+1)))

        # Epsilon driven dynamic noise exploration.
        # action = np.argmax(Q[state]) if np.random.rand() > epsilon else np.argmax(Q[state] + np.random.randn(1, env.action_space.n))

        # Epsilon driven.
        action = np.argmax(Q[np.array([state])[np.newaxis].T]) if np.random.rand() > epsilon else env.action_space.sample()

        n_state, reward, done, info = env.step(action)

        Q[state, action] += alpha * (reward + GAMMA * np.max(Q[n_state]) - Q[state, action])

        total_reward += reward

        if done:
            # print("Episode finished after {} timesteps".format(frame+1))
            break

        state = n_state

    alpha = max(ALPHA_MIN, alpha-ALPHA_MAX)
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
