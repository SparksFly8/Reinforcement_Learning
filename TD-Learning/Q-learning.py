import random


def init_Q_table(env):
    '''初始化Q表'''
    Q = {env.STATE_A:{action:0 for action in range(env.nA)},
         env.STATE_B:{action:0 for action in range(env.nB)},
         env.Terminal:{action:0 for action in range(env.nA)}}
    return Q

def select_action_behavior_policy(action_value_dict, epsilon):
    '''使用epsilon-greedy采样action'''
    if random.random() > epsilon:
        max_keys = [key for key, value in action_value_dict.items() if value == max( action_value_dict.values() )]
        action = random.choice(max_keys)
    else:
        # 从Q字典对应state中随机选取1个动作,由于返回list,因此通过[0]获取元素
        action = random.sample(action_value_dict.keys(), 1)[0]
    return action

def Q_learning(env, alpha=0.2, epsilon_scope=[0.2,0.05,0.99], num_of_episode=1000, gamma=0.9):
    '''
    Q学习算法,返回Q表和估计的最优策略
    其中epsilon_scope由高到低衰减,从左到右分别是[最高值,最低值,衰减因子]
    '''
    epsilon = epsilon_scope[0]
    # 1. 初始化Q表
    Q = init_Q_table(env)
    for num in range(num_of_episode):
        state = env.reset()  # Init S
        while True:
            # 2.通过behavior policy采样action
            action = select_action_behavior_policy(Q[state], epsilon)
            # 3.执行action并观察R和next state
            next_state, reward, done = env.step(action)
            # 4.更新Q(S,A),使用max操作更新
            Q[state][action] += alpha * (reward + gamma*max( Q[next_state].values() ) - Q[state][action])  
            if done: break
            state = next_state
        # 对epsilon进行衰减
        if epsilon >= epsilon_scope[1]: epsilon *= epsilon_scope[2]
    return Q


env = Env(-0.1, 1, 10)   
# Q-learning学习出Q表
Q_table = Q_learning(env, alpha=0.2, epsilon_scope=[0.2,0.05,0.99], num_of_episode=300, gamma=0.9)  
Q_table