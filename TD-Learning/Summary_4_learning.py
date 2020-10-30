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


def TD_learning(env, method='Q-Learning', alpha=0.2, epsilon_scope=[0.2, 0.05, 0.99], num_of_episode=1000, gamma=0.9):
    '''
    TD学习算法,返回Q表和估计的最优策略
    其中epsilon_scope由高到低衰减,从左到右分别是[最高值,最低值,衰减因子]
    '''
    epsilon = epsilon_scope[0]
    # 1. 初始化Q1表和Q2表
    Q = init_Q_table(env)
    if method == 'Double-Q':
        Q2 = init_Q_table(env)
    bool_A_left = np.zeros(num_of_episode)
    Aleft_Q_values = []
    B_max_Q_values = []
    for num in range(num_of_episode):
        state = env.reset()  # Init S
        while True:
            # 2.通过behavior policy采样action
            if method == 'Double-Q':
                add_Q1_Q2_state = {action: Q1_value + Q2[state][action] for action, Q1_value in Q[state].items()}
                action = select_action_behavior_policy(add_Q1_Q2_state, epsilon)
            else:
                action = select_action_behavior_policy(Q[state], epsilon)
            if state == env.STATE_A and action == env.left:
                bool_A_left[int(num)] += 1
            # 3.执行action并观察R和next state
            next_state, reward, done = env.step(action)
            # 4.更新Q(S,A),使用max操作更新
            if method == 'Q-Learning':
                Q[state][action] += alpha * (reward + gamma * max(Q[next_state].values()) - Q[state][action])
            elif method == 'Expected_Sarsa':
                Q[state][action] += alpha * (
                            reward + gamma * sum(Q[next_state].values()) / len(Q[next_state]) - Q[state][action])
            elif method == 'Action_Distribution':
                Q[state][action] += alpha * (
                            reward + gamma * random.choice(list(Q[next_state].values())) - Q[state][action])
            elif method == 'Double-Q':
                if random.random() >= 0.5:
                    # 从Q1表中的下一步state找出状态价值最高对应的action视为Q1[state]的最优动作
                    A1 = random.choice(
                        [action for action, value in Q[next_state].items() if value == max(Q[next_state].values())])
                    # 将Q1[state]得到的最优动作A1代入到Q2[state][A1]中的值作为Q1[state]的更新
                    Q[state][action] += alpha * (reward + gamma * Q2[next_state][A1] - Q[state][action])
                else:
                    A2 = random.choice(
                        [action for action, value in Q2[next_state].items() if value == max(Q2[next_state].values())])
                    Q2[state][action] += alpha * (reward + gamma * Q[next_state][A2] - Q2[state][action])
            if done: break
            state = next_state

        Aleft_Q_values.append(Q[env.STATE_A][env.left])
        B_max_Q_values.append(max(Q[env.STATE_B].values()))
        # 对epsilon进行衰减
        if epsilon >= epsilon_scope[1]: epsilon *= epsilon_scope[2]
        # if num % 20 == 0:  print("Episode: {}, Score: {}".format(num, sum_reward))
    return Q, bool_A_left, Aleft_Q_values, B_max_Q_values


# method = ['Q-Learning', 'Expected_Sarsa', 'Action_Distribution', 'Double-Q']
env = Env(-0.1, 1, 10)
# Q-learning学习出Q表
Q_table, _, _, _ = TD_learning(env, method='Double-Q', alpha=0.2, epsilon_scope=[0.2, 0.05, 0.99], num_of_episode=300,
                               gamma=0.9)
Q_table