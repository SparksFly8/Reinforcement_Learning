def show_figure(prob_Q_A_left, prob_E_A_left, prob_AD_A_left, prob_Q2_A_left):
    import matplotlib.pyplot as plt
    plt.ylabel('% left actions from A')
    plt.xlabel('Episodes')
    x_ticks = np.arange(0, 301, 20)
    y_ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks, ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.plot(range(300), prob_Q_A_left, '-', label='Q Learning')
    plt.plot(range(300), prob_E_A_left, '-', label='Double Q-Learning')
    plt.plot(range(300), prob_AD_A_left, '-', label='Action Distribution')
    plt.plot(range(300), prob_Q2_A_left, '-', label='Expected Sarsa')
    plt.plot(np.ones(300) * 0.05, label='Optimal')
    plt.title('Comparison of the effect of 4 algorithms on Ex 6.7')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()


total_num = 1000

A_Q_lst, B_Q_lst = np.zeros((total_num, 300)), np.zeros((total_num, 300))
A_Q2_lst, B_Q2_lst = np.zeros((total_num, 300)), np.zeros((total_num, 300))
A_AD_lst, B_AD_lst = np.zeros((total_num, 300)), np.zeros((total_num, 300))
A_E_lst, B_E_lst = np.zeros((total_num, 300)), np.zeros((total_num, 300))

prob_Q_A_left = np.zeros((total_num, 300))
prob_Q2_A_left = np.zeros((total_num, 300))
prob_AD_A_left = np.zeros((total_num, 300))
prob_E_A_left = np.zeros((total_num, 300))
# 计算在STATE_A下采样动作left的概率

alpha = 0.1
start_epsilon = 0.1
gamma = 0.9
num_of_episode = 300

for num in tqdm(range(total_num)):
    _, A_left1, A_Q1, B_Q1 = TD_learning(env, 'Q-Learning', alpha, epsilon_scope=[start_epsilon, 0.05, 1],
                                         num_of_episode=num_of_episode, gamma=gamma)
    _, A_left2, A_Q2, B_Q2 = TD_learning(env, 'Double-Q', alpha, epsilon_scope=[start_epsilon, 0.05, 1],
                                         num_of_episode=num_of_episode, gamma=gamma)
    _, A_left3, A_Q3, B_Q3 = TD_learning(env, 'Action_Distribution', alpha, epsilon_scope=[start_epsilon, 0.05, 1],
                                         num_of_episode=num_of_episode, gamma=gamma)
    _, A_left4, A_Q4, B_Q4 = TD_learning(env, 'Expected_Sarsa', alpha, epsilon_scope=[start_epsilon, 0.05, 1],
                                         num_of_episode=num_of_episode, gamma=gamma)

    prob_Q_A_left[int(num)] = A_left1
    prob_Q2_A_left[int(num)] = A_left2
    prob_AD_A_left[int(num)] = A_left3
    prob_E_A_left[int(num)] = A_left4

    A_Q_lst[int(num)], B_Q_lst[int(num)] = A_Q1, B_Q1
    A_Q2_lst[int(num)], B_Q2_lst[int(num)] = A_Q2, B_Q2
    A_AD_lst[int(num)], B_AD_lst[int(num)] = A_Q3, B_Q3
    A_E_lst[int(num)], B_E_lst[int(num)] = A_Q4, B_Q4

a = prob_Q_A_left.mean(axis=0)
b = prob_Q2_A_left.mean(axis=0)
c = prob_AD_A_left.mean(axis=0)
d = prob_E_A_left.mean(axis=0)

show_figure(a, b, c, d)