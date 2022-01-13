from matplotlib.colors import Colormap
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# somewhat rewritten function from the internet to find perpendicular vectors
def get_perpendicular_vector(w):
    if w.shape[0] == 2:
        new_v = [w[1], -w[0]]
        return new_v
    else:
        second_v = np.zeros(w.shape[0])
        second_v[random.randint(0, w.shape[0]-1)] = 1
        return np.cross(w, second_v)

def generate_random_data(number_of_observations, N):
    mean = 0
    variance = 1
    sd = variance ** 0.5
    x = np.random.normal(loc=mean, scale=sd, size=(number_of_observations, N))
    y = np.array([-1 if random.random() < 0.5 else 1 for _ in range(number_of_observations)])
    return x, y

def train(x, y, max_epochs):
    success = 0
    w = np.zeros(len(x[0]))
    for epoch in range(max_epochs):
        updated = False
        for example_index in range(len(x)):
            E_mu = np.dot(w, (x[example_index] * y[example_index]))
            N = len(x[example_index])
            if E_mu <= 0:
                updated = True
                update = 1/N * x[example_index] * y[example_index]
                w += update
        if not updated:
            success = 1
            break
    return success

def make_plot(x, ys, save=False, title='Linear Separability of Datasets', label_desc = 'N: '):
    plt.figure()
    plt.title(title)
    plt.ylim(-0.1,1.1)
    plt.xlabel('P/N (alpha)')
    plt.ylabel('p(linearly seperable)')
    for i, y in ys.items():
        plt.plot(x, y, label=f'{label_desc}{i}')
    plt.legend()
    if save:
        plt.savefig(f'plots/A1_plots/{title}.jpg')

    plt.show()

def main():
    alphas = []
    all_fractions = {}
    epoch_sizes = [100]
    # Ns = [5, 25, 50, 100, 200, 500]
    Ns = [5,]

    alphas = np.arange(0.7, 3.01, 0.1)
    total_datasets = 50
    for max_epochs in tqdm(epoch_sizes, desc='running different max epochs', leave=False):
        for N in tqdm(Ns, desc='running different sizes of N', leave=False):
            fractions = []
            for alpha in tqdm(alphas, desc='running different alphas', leave=False):
                input_size = int(alpha * N)

                num_seperable = 0

                for n in tqdm(range(0, total_datasets), desc='training perceptron on different datasets', leave=False):
                    x_data, y_data = generate_random_data(input_size, N)
                    success = train(x_data, y_data, max_epochs)
                    num_seperable += success

                fraction_Q = (num_seperable / total_datasets)
                fractions.append(fraction_Q)
            all_fractions[N] = fractions
    print(all_fractions)
    make_plot(alphas, all_fractions, save=False,)



if __name__ =='__main__':
    main()
