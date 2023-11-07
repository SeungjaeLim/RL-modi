import numpy as np

class Qlearning_updater():
    def __init__(self, env, epsilon, gamma, lr):
        super().__init__()
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    def action_strategy(self, state):
        action=0
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        return action

    def Bellman_Q_updater(self, state, state2, reward, action):
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = self.Q[state, action] + self.lr * (target - predict)

        return self.Q

