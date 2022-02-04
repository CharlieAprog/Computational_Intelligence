from unicodedata import normalize
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def generate_teacher(N):
    '''
    function that will generate a weight vector so that its norm^2 will be N
    '''
    w_random = np.random.normal(size=(N))
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
    y = np.array([np.sign(np.dot(element, w_teacher)) for element in x])
    return x, y, w_teacher

def determine_stabilities(w, x, y):
    norm = np.linalg.norm(w)
    return np.array([np.dot(w, x[i] * y[i]) / norm for i in range(len(y))])

def get_lowest_stability_index(k):
    return np.argmin(k)

def stopping_criterion(prev_stability, current_stability):
    '''
    funcation that determines when the training will stop
    returns true when conditions for stopping are met
    returns false otherwise
    '''
    return np.array_equal(prev_stability, current_stability)


def train(x, y, max_epochs):
    N = len(x[0])
    prev_stability = np.empty((1,N), dtype=float)
    w = np.zeros((1,N))

    for epoch in range(max_epochs):
        k = determine_stabilities(w, x, y)
        i = get_lowest_stability_index(k)
        w += 1/N * (x[i] * y[i])
        if stopping_criterion(prev_stability, k[i]):
            break
        prev_stability = k[i]
    return w, k

def make_plot(x, ys, save=False, title='Generalisation Error of Algorithm', label_desc = 'N: '):
    plt.figure()
    plt.title(title)
    plt.ylim(-0.1,1.1)
    plt.xlabel('P/N (alpha)')
    plt.ylabel('Generalisaition Error')
    for i, y in ys.items():
        plt.plot(x, y, label=f'{label_desc}{i}')
    plt.legend()
    if save:
        plt.savefig(f'plots/A2_plots/{title}.jpg')
    plt.show()

def calc_generalisation_error(w, w_teacher):
    calc = np.dot(w, w_teacher)/ (np.linalg.norm(w) * np.linalg.norm(w_teacher))
    return 1 / math.pi * np.arccos(calc)

def plot_stabilities(stabilities, title='Distribution of stabilities', save= False):
    hist_arr = []
    for ks in stabilities:
        for k in ks:
            hist_arr.append(k[0])
    plt.figure()
    plt.title(title)
    plt.xlabel('K, Distance from the hyperplane')
    plt.ylabel('Amount of data points')
    #plt.legend()
    plt.hist(hist_arr, bins=30, range=(0, 5))
    if save:
        plt.savefig(f'plots/A2_plots/{title}.jpg')
    plt.show()

def main():
    total_datasets = 20
    all_errors = {}

    Ns = [5, 50, 100]
    epoch_sizes = [300]
    alphas = np.arange(0.25, 5.01, 0.25)
    
    for max_epochs in tqdm(epoch_sizes, desc='running different sizes of N', leave=False):
        for N in tqdm(Ns, desc='running different sizes of N', leave=False):
            errors = []
            stabilities = []
            for alpha in tqdm(alphas, desc='running different alphas', leave=False):
                P = int(alpha * N)
                generalisation = 0
                for n in tqdm(range(0, total_datasets), desc='training perceptron on different datasets', leave=False):
                    x_data, y_data, w_teacher = generate_random_data(P, N)
                    w_max, k = train(x_data, y_data, max_epochs)
                    generalisation += calc_generalisation_error(w_max, w_teacher)
                    if alpha == 5:
                        stabilities.append(k)
                average_generalisation_error = generalisation / total_datasets
                errors.append(average_generalisation_error)
            all_errors[N] = errors
    plot_stabilities(stabilities, save=True)
    make_plot(alphas, all_errors, save=True)


if __name__ == '__main__':
    main()
