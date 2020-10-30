import numpy as np


class Env():
    '''构造一个环境类'''

    def __init__(self, mu, sigma, nB):
        self.mu = mu
        self.sigma = sigma
        self.STATE_A = self.left = 0
        self.STATE_B = self.right = 1
        self.Terminal = 2
        self.nS = 3  # 加上Terminal即3个状态
        self.nA = 2
        self.nB = nB  # 状态B的动作数
        self.state = self.STATE_A

    def reset(self):
        self.state = self.STATE_A
        return self.state

    def step(self, action):
        # A--left
        if self.state == self.STATE_A and action == self.left:
            self.state = self.STATE_B
            return self.state, 0, False  # next_state, reward, done
        # A--right
        elif self.state == self.STATE_A and action == self.right:
            self.state = self.Terminal
            return self.state, 0, True
        # B--all_actions
        elif self.state == self.STATE_B:
            self.state = self.Terminal
            reward = random.normalvariate(self.mu, self.sigma)
            return self.state, reward, True