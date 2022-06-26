import gym
import numpy as np

class DiscreteActionWrapper1d(gym.ActionWrapper):

    def __init__(self, env, n_discrete_actions):

        super(DiscreteActionWrapper1d, self).__init__(env)

        self.state_space = env.observation_space
        self.n_discrete_actions = n_discrete_actions
        self.action_space = gym.spaces.Discrete(n_discrete_actions)
        self.action_vectors = np.linspace(self.env.action_space.low.item(),self.env.action_space.high.item(),num=n_discrete_actions)
        self.true_action_vectors = np.eye(n_discrete_actions)
        self.a_max = 1

    def action(self, action):

       cts_action = np.array([self.action_vectors[action]])

       return cts_action