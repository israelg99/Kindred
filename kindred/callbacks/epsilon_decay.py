from rl.callbacks import Callback

class EpsilonDecay(Callback):
    def __init__(self, decay, minimum=0.01, offset=0, skip=1):
        super().__init__()
        self.decay = decay
        self.min = minimum
        self.offset = offset
        self.skip = skip

    def on_episode_end(self, episode, logs={}):
        super().on_episode_end(episode, logs=logs)
        episode += 1
        if episode >= self.offset and episode % self.skip == 0:
            self.model.policy.eps = max(self.min, self.model.policy.eps*self.decay)

        print("epsilon:", self.model.policy.eps)
