import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


# k-armed
config = {
    "seed": 11,  # ramdom seed for numpy
    "k": 10,  # k-armed bandit problem
    "step": 100000,  # steps for k-armed bandit problem
    "epsilon": 0.1,
}





def k_armed_bandit_avg_sample(k, step, epsilon, bandit_change=False):

    bandit = [np.random.normal(0, 1) for i in range(k)]
    # print(f"bandit({k}) is \n{bandit}")
    Qa = [0 for i in range(k)]
    Na = [0 for i in range(k)]

    avg_rwd = 0
    hist_avg_rwd = []

    for i in tqdm(range(step), desc=f'nonstationary, epsilon={epsilon}'):
        if np.random.random() < epsilon:
            # randomly chose one
            idx = np.random.randint(0, k)
        else:
            # chose argmax Qa
            idx = np.argmax(Qa)

        if bandit_change:
            # v = np.random.normal(0, 0.5)
            for j in range(k):
                bandit[j] += np.random.normal(0, 0.01)
                # bandit[j] += v
        
        R = bandit[idx]

        Na[idx] += 1
        Qa[idx] = Qa[idx] + (1 / Na[idx])*(R - Qa[idx])

        avg_rwd = avg_rwd + 1/(i+1)*(R - avg_rwd)
        hist_avg_rwd.append(avg_rwd)
    
    return hist_avg_rwd


def k_armed_bandit_constant_step_size(k, step, epsilon, alpha, bandit_change=False):
    bandit = [np.random.normal(0, 1) for i in range(k)]
    # print(f"bandit({k}) is \n{bandit}")
    Qa = [0 for i in range(k)]
    Na = [0 for i in range(k)]

    avg_rwd = 0
    hist_avg_rwd = []

    for i in tqdm(range(step), desc=f'stationary, epsilon={epsilon}'):
        if np.random.random() < epsilon:
            # randomly chose one
            idx = np.random.randint(0, k)
        else:
            # chose argmax Qa
            idx = np.argmax(Qa)

        if bandit_change:
            # v = np.random.normal(0, 0.5)
            for j in range(k):
                bandit[j] += np.random.normal(0, 0.01)
                # bandit[j] += v
        
        R = bandit[idx]

        Na[idx] += 1
        Qa[idx] = Qa[idx] + alpha*(R - Qa[idx])

        avg_rwd = avg_rwd + 1/(i+1)*(R - avg_rwd)
        hist_avg_rwd.append(avg_rwd)
    
    return hist_avg_rwd

def main():
    np.random.seed(config["seed"])

    # plt.subplot(211)
    # for i in np.arange(0, 1, 0.1):
    #     avg_rwd = k_armed_bandit_avg_sample(10, config['step'], i, True)
    #     plt.plot(avg_rwd, label=f'{i}')
    # plt.legend()
    # plt.title('avrage smaple')

    # plt.subplot(212)
    # for i in np.arange(0, 1, 0.1):
    #     avg_rwd = k_armed_bandit_constant_step_size(10, config['step'], i, 0.1, True)
    #     plt.plot(avg_rwd, label=f'{i}')
    # plt.legend()
    # plt.title('fixed-step size')

    plt.show()

    avg_rwd = k_armed_bandit_avg_sample(10, config['step'], epsilon=0.1, bandit_change=True)
    plt.plot(avg_rwd, label=f'avrage sample')
    avg_rwd = k_armed_bandit_constant_step_size(10, config['step'], epsilon=0.1, alpha=0.9, bandit_change=True)
    plt.plot(avg_rwd, label=f'fixed-step size')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()

