import random

def n_step_TD_prediction(env, n=4, alpha=0.4, num_of_episode=10, gamma=1):
    '''n-step TD预测,返回state value表'''
    # 1.Init V(S)
    value = np.zeros(env.nS)
    for episode in range(num_of_episode):
        env.reset()
        # 1.1 Init S_set and R_set and store S0 and R0
        state_lst = [env.START_STATE]
        reward_lst = [0]
        # 总时间步(火车头)
        t = 0
        T = float('inf')
        while True:
            t += 1
            # 2.采样并存储所有时间步下的reward和state
            if t < T:
                # 2.1 等概率采样并执行动作,得到下一步信息
                action = env.sample_action()
                next_state, reward, done = env.step(action)
                # 2.2 存储next_state和reward
                reward_lst.append(reward)
                state_lst.append(next_state)
                # 2.3 探索到terminal状态后就不再采样动作,第2部分代码块不会再执行
                if done:
                    T = t
            # 更新时间步(火车尾)
            update_t = t - n
            # 3.计算n-step内的累计reward和V(St+n)得到的回报returns，然后更新V(St)
            if update_t >= 0:
                returns = 0
                # 3.1 计算n-step之间的累计reward
                for time in range( update_t + 1, min(update_t + n, T) + 1 ):
                    returns += gamma**(time - update_t - 1) * reward_lst[time]
                # 3.2 与V(St+n)累加得到完整returns
                if update_t + n < T:
                    returns += gamma**n * value[ state_lst[update_t + n] ]
                # 3.3 更新V(St)
                value[ state_lst[update_t] ] += alpha * (returns - value[ state_lst[update_t] ])
            # 更新到terminal前一步则退出循环
            if update_t == T - 1:
                break
    return value