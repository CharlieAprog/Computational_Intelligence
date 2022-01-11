from unicodedata import normalize
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_teacher(N):
    '''
    function that will generate a weight vector so that its norm^2 will be N
    '''
    w_random = np.random.normal(size=(1, N))
    norm = np.linalg.norm(w_random)
    return (N ** 0.5 / norm) * w_random

def generate_random_data(P, N):
    '''
    function that will generate a dataset and a set of lables
    the lables are determined by a teacher weight (the solution)
    '''
    mean = 0
    variance = 1
    sd = variance ** 0.5
    w_teacher = generate_teacher(N)
    x = np.random.normal(loc=mean, scale=sd, size=(P, N))
    y = np.array([element * w_teacher for element in x])
    return x, y

def determine_stabilities(w, x, y):
    norm = np.linalg.norm(w)
    norm = norm if norm != 0.0 else 0.01
    print('\n')
    print(y[0], x[0], norm)

    print((w * (y[0] * x[0])) / norm)
    return np.array([(w * (y[i] * x[i])) / norm for i in range(len(y))])

def get_lowest_stability_index(k):
    return np.argmin(k)


def train(x, y, max_epochs):
    success = 0
    N = len(x[0])
    prev_stability = -1
    w = np.zeros((1,N))

    for epoch in range(max_epochs):
        k = determine_stabilities(w, x, y)
        i = get_lowest_stability_index(k)
        print('\n')
        print(w.shape)
        print((x[i] * y[i]).shape)
        print(k[i])
        w += 1/N * (x[i] * y[i])
        if prev_stability == k[i]:
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
        plt.savefig(f'plots/{title}.jpg')
    plt.show()

def main():
    total_datasets = 50
    all_fractions = {}

    dimension_sizes = [100]
    epoch_sizes = [300]
    alphas = np.arange(0.7, 3.01, 0.1)
    for max_epochs in tqdm(epoch_sizes, desc='running different sizes of N', leave=True):
        for dimension_size in tqdm(dimension_sizes, desc='running different sizes of N', leave=True):
            fractions = []
            for alpha in tqdm(alphas, desc='running different alphas', leave=False):
                input_size = int(alpha * dimension_size)
                num_seperable = 0

                for n in tqdm(range(0, total_datasets), desc='training perceptron on different datasets', leave=False):
                    x_data, y_data = generate_random_data(input_size, dimension_size)
                    success = train(x_data, y_data, max_epochs)
                    num_seperable += success
                print(num_seperable)
                fraction_Q = (num_seperable / total_datasets)
                fractions.append(fraction_Q)
            all_fractions[max_epochs] = fractions


if __name__ == '__main__':
    main()
