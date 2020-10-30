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

def get_Q1_add_Q2(Q1_state_dict, Q2_state_dict):
    '''返回Q1[state]+Q2[state]'''
    return {action: Q1_value + Q2_state_dict[action] for action, Q1_value in Q1_state_dict.items()}

def double_Q_learning(env, alpha=0.2, epsilon_scope=[0.2,0.05,0.99], num_of_episode=1000, gamma=0.9):
    '''
    双Q学习算法,返回Q表和估计的最优策略
    其中epsilon_scope由高到低衰减,从左到右分别是[最高值,最低值,衰减因子]
    '''
    epsilon = epsilon_scope[0]
    # 1. 初始化Q1表和Q2表
    Q1 = init_Q_table(env)
    Q2 = init_Q_table(env)
    for num in range(num_of_episode):
        state = env.reset()  # Init S
        while True:
            # 2.通过behavior policy采样action
            add_Q1_Q2_state = get_Q1_add_Q2(Q1[state], Q1[state])
            action = select_action_behavior_policy(add_Q1_Q2_state, epsilon)
            # 3.执行action并观察R和next state
            next_state, reward, done = env.step(action)
            # 4.更新Q(S,A),使用max操作更新
            if random.random() >= 0.5:
                # 从Q1表中的下一步state找出状态价值最高对应的action视为Q1[state]的最优动作
                A1 = random.choice( [action for action, value in Q1[next_state].items() if value == max( Q1[next_state].values() )] )
                # 将Q1[state]得到的最优动作A1代入到Q2[state][A1]中的值作为Q1[state]的更新
                Q1[state][action] += alpha * (reward + gamma*Q2[next_state][A1] - Q1[state][action])
            else:
                A2 = random.choice( [action for action, value in Q2[next_state].items() if value == max( Q2[next_state].values() )] )
                Q2[state][action] += alpha * (reward + gamma*Q1[next_state][A2] - Q2[state][action])
            if done: break
            state = next_state
        # 对epsilon进行衰减
        if epsilon >= epsilon_scope[1]: epsilon *= epsilon_scope[2]
    return Q1


env = Env(-0.1, 1, 10)
# Q-learning学习出Q表
Q_table = double_Q_learning(env, alpha=0.2, epsilon_scope=[0.2,0.05,0.99], num_of_episode=300, gamma=0.9)
Q_table