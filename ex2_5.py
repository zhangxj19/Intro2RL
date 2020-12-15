import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


# k-armed
config = {
    "seed": 41,  # ramdom seed for numpy
    "k": 10,  # k-armed bandit problem
    "step": 1000,  # steps for k-armed bandit problem
    "epsilon": 0.1,
    # "exp": 1, # 1 for exploit epsilon for average sample meethod
    "exp": 2, # 2 for comparasion of avgrage sample and constant step size
}

def k_armed_bandit_avg_sample(k, step, epsilon, bandit, bandit_change=False):
    # bandit = [np.random.normal(0, 1) for i in range(k)]
    # print(f"bandit({k}) is \n{bandit}")
    Qa = [0 for i in range(k)]
    Na = [0 for i in range(k)]

    avg_rwd = 0
    hist_avg_rwd = []

    for i in tqdm(range(step), desc=f'average sample, epsilon={epsilon}'):
        if np.random.random() < epsilon:
            # randomly chose one
            idx = np.random.randint(0, k)
        else:
            # chose argmax Qa
            idx = np.argmax(Qa)

        if bandit_change:
            v = np.random.normal(0, 0.01)
            for j in range(k):
                # bandit[j] += np.random.normal(0, 0.01)
                bandit[j] += v
        
        R = bandit[idx]

        Na[idx] += 1
        Qa[idx] = Qa[idx] + (1 / Na[idx])*(R - Qa[idx])

        avg_rwd = avg_rwd + 1/(i+1)*(R - avg_rwd)
        hist_avg_rwd.append(avg_rwd)
    
    return hist_avg_rwd


def k_armed_bandit_constant_step_size(k, step, epsilon, alpha, bandit, bandit_change=False):
    # bandit = [np.random.normal(0, 1) for i in range(k)]
    # print(f"bandit({k}) is \n{bandit}")
    Qa = [0 for i in range(k)]
    Na = [0 for i in range(k)]

    avg_rwd = 0
    hist_avg_rwd = []

    for i in tqdm(range(step), desc=f'constant step, epsilon={epsilon}'):
        if np.random.random() < epsilon:
            # randomly chose one
            idx = np.random.randint(0, k)
        else:
            # chose argmax Qa
            idx = np.argmax(Qa)

        if bandit_change:
            v = np.random.normal(0, 0.01)
            for j in range(k):
                # bandit[j] += np.random.normal(0, 0.01)
                bandit[j] += v
        
        R = bandit[idx]

        Na[idx] += 1
        Qa[idx] = Qa[idx] + alpha*(R - Qa[idx])

        avg_rwd = avg_rwd + 1/(i+1)*(R - avg_rwd)
        hist_avg_rwd.append(avg_rwd)
    
    return hist_avg_rwd


def main():
    np.random.seed(config["seed"])
    bandit = [np.random.normal(0, 1) for i in range(config['k'])]

    if config['exp'] == 1:

        plt.subplot(211)
        for i in np.arange(0, 1, 0.1):
            avg_rwd = k_armed_bandit_avg_sample(10, config['step'], epsilon=i, bandit=bandit[:], bandit_change=True)
            plt.plot(avg_rwd, label=f'{i}')
        plt.legend()
        plt.title('avrage smaple')

        plt.subplot(212)
        for i in np.arange(0, 1, 0.1):
            avg_rwd = k_armed_bandit_constant_step_size(10, config['step'], epsilon=i, alpha=0.1, bandit=bandit[:], bandit_change=True)
            plt.plot(avg_rwd, label=f'{i}')
        plt.legend()
        plt.title('fixed-step size')

        plt.show()
    elif config['exp'] == 2:
        config['step'] = 10000
        avg_rwd = k_armed_bandit_avg_sample(10, config['step'], epsilon=0.1, bandit=bandit[:], bandit_change=True)
        plt.plot(avg_rwd, label=f'avrage sample')
        avg_rwd = k_armed_bandit_constant_step_size(10, config['step'], epsilon=0.1, alpha=0.1, bandit=bandit[:], bandit_change=True)
        plt.plot(avg_rwd, label=f'fixed-step size')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    main()

