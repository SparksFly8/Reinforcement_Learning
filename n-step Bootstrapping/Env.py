import random
import numpy as np


class Env():
    '''构造一个环境类'''

    def __init__(self, num_states=19):
        self.STATES = np.arange(1, num_states + 1)
        self.Terminal = [0, num_states + 1]
        self.START_STATE = (1 + num_states) // 2  # 19 to 10
        self.left, self.right = 0, 1
        self.action_space = [self.left, self.right]
        self.nS = num_states + 2  # 加上Terminal的两个状态即21个状态
        self.nA = 2
        self.state = self.START_STATE

    def reset(self):
        self.state = self.START_STATE
        return self.state

    def sample_action(self):
        '''等概率选择动作'''
        if random.random() > 0.5:
            return self.left
        else:
            return self.right

    def step(self, action):
        if self.state in self.Terminal:
            self.state = self.START_STATE
        done = False
        reward = 0
        # left-action
        if action == self.left:
            self.state -= 1
            if self.state in self.Terminal:
                done = True
                reward = -1
        # right-action
        else:
            self.state += 1
            if self.state in self.Terminal:
                done = True
                reward = 1
        return self.state, reward, done