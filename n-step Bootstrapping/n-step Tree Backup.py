import random

def policy(action_value_set, action, epsilon=0.2):
    '''返回当前state下选择该action的概率'''
    best_action = np.argmax(action_value_set)
    if best_action == action:
        return 1 - epsilon
    return epsilon

def n_step_TreeBackup(env, n=4, alpha=0.4, epsilon_scope=[0.2,0.05,0.99], num_of_episode=10, gamma=1):
    '''n-step Sarsa control,返回Q表'''
    epsilon = epsilon_scope[0]
    # 1.Init Q(s,a)
    Q = np.zeros( (env.nS, env.nA) )
    for _ in range(num_of_episode):
        env.reset()
        # 1.1 Init S_set, A_set and R_set and store S0, A0 and R0
        state_lst = [env.START_STATE]
        reward_lst = [0]
        # 1.2 等概率随机选择并存储A0
        action = env.sample_action()
        action_lst = [action]
        # 总时间步(火车头)
        t = 0
        T = float('inf')
        while True:
            t += 1
            # 2.采样并存储所有时间步下的reward和state
            if t < T:
                # 2.1 执行动作,得到下一步信息
                next_state, reward, done = env.step(action)
                # 2.2 存储next_state和reward
                reward_lst.append(reward)
                state_lst.append(next_state)
                # 2.3 等概率随机采样下一步动作,若探索到terminal状态后则不再采样动作,第2部分代码块不会再执行
                if done: T = t
                else:
                    action = env.sample_action()
                    action_lst.append(action)
            # 更新时间步(火车尾)
            update_t = t - n
            # 3.计算n-step内的累计reward和Q(Sτ+n,Aτ+n)得到的回报returns，然后更新Q(Sτ,Aτ)
            if update_t >= 0:
                # 3.1 计算叶结点G_{t+n-1:t+n}
                returns = reward
                if t < T:
                    # G = Rt+1 + γ.∑_{a}.π(a|St+1).Q(St+1,a)
                    for act in env.action_space:
                        returns += gamma * policy(Q[next_state],act,epsilon) * Q[next_state][act]
                # 3.2 递归从叶结点回溯到当前节点计算出G_{t:t+n}
                for time in range( min(t, T), update_t, -1):
                    Sk = state_lst[time]
                    # expect = γ.∑_{a≠At}.π(a|Sk).Q(Sk,a) + γ.π(Ak|Sk).G
                    expect = 0
                    for Ak in env.action_space:
                        if Ak == action:
                            # γ.π(Ak|Sk).G
                            expect += gamma * policy(Q[Sk],action,epsilon) * returns
                        else:
                            # γ.∑_{a≠At}.π(a|Sk).Q(Sk,a)
                            expect += gamma * policy(Q[Sk],Ak,epsilon) * Q[Sk][Ak]
                    # G = Rk + γ.∑_{a≠Ak}.π(a|Sk).Q(Sk,a) + γ.π(Ak|Sk).G
                    returns = reward_lst[time] + expect
                # 3.3 更新Q(Sτ,Aτ)
                Q[state_lst[update_t]][action_lst[update_t]] += alpha * (returns - Q[state_lst[update_t]][action_lst[update_t]])
            # 更新到terminal前一步则退出循环
            if update_t == T - 1:
                break
        # 对epsilon进行衰减
        if epsilon >= epsilon_scope[1]: epsilon *= epsilon_scope[2]
    return Q