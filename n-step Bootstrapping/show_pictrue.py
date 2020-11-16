def get_average_error(env):
    TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
    TRUE_VALUE[0] = TRUE_VALUE[-1] = 0

    steps = np.power(2, np.arange(0, 10))
    alphas = np.arange(0, 1.1, 0.1)
    num_of_episode = 10
    runs = 100
    gamma = 1

    errors = np.zeros((len(steps), len(alphas)))
    for run in range(runs):
        for step_i, n in enumerate(steps):
            for alpha_i, alpha in enumerate(alphas):
                value = n_step_TD_prediction(env, n, alpha, num_of_episode, gamma)
                errors[step_i, alpha_i] += np.sqrt(np.sum(np.power(value - TRUE_VALUE, 2)) / 19)
        print(
            "\r" + "#" * round((int(run) + 1) / runs * 60) + " " * (60 - round((int(run) + 1) / runs - 1)) + '|' + str(
                int(run) + 1) + '/' + str(runs), sep="", end="")
    errors /= runs
    return errors

def show(errors):
    steps = np.power(2, np.arange(0, 10))
    alphas = np.arange(0, 1.1, 0.1)
    for i in range(0, len(steps)):
        plt.plot(alphas, errors[i, :], label='n = %d' % (steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('Average RMS error over 19 states and 10 episodes')
    plt.ylim([0.15, 0.65])
    plt.title('Performance of n-step TD methods as a fun of alpha')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()


errors = get_average_error(env)
show(errors)