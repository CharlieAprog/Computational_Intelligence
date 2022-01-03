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

def generate_random_data(number_of_observations, dimension_size):
    mean = 0
    variance = 1
    sd = variance ** 0.5
    x = np.random.normal(loc=mean, scale=sd, size=(number_of_observations, dimension_size))
    y = np.array([-1 if random.random() < 0.5 else 1 for _ in range(number_of_observations)])
    return x, y

def train(x, y, w, max_epochs):
    for epoch in range(max_epochs):
        updated = False
        for example_index in range(len(x)):
            E_mu = np.dot(w, (x[example_index] * y[example_index]))
            if E_mu <= 0:
                updated = True
                update = 1/len(x[example_index]) * x[example_index] * y[example_index]
                w += update
        if not updated:
            break
    return w

def count_ls(w, x, y, number_of_observations):
    lin_sep_count = 0
    for example in range(number_of_observations):
        label = np.dot(w, (x[example] * y[example]))
        if label > 0:
            lin_sep_count += 1
    return lin_sep_count


def make_plot(x, y, save=False):
    title = f'Capacity of Model'
    plt.figure()
    plt.title(title)
    plt.ylim(0,1)
    plt.set_xlabel('P/N (alpha)')
    plt.set_ylabel('p(linearly seperable)')
    plt.plot(x, y)
    if save:
        plt.savefig(f'plots', title)
    plt.show()



def main():
    alphas = []
    fractions = []
    for alpha in tqdm(np.arange(0.5, 5.1, 0.25), desc='running different alphas', leave=True):
        # parameters
        N = 100
        P = int(alpha * N)
        max_epochs = 100
        w = np.zeros(N)
        num_seperable = 0
        total = 0

        for n in tqdm(range(0, 50), desc='running values of N', leave=False):
            data_x, data_y = generate_random_data(P, N)
            w = train(data_x, data_y, w, max_epochs)
            num_seperable += count_ls(w, data_x, data_y, P)
            total += P
        fraction_Q = (num_seperable / total)
        alphas.append(alpha)
        fractions.append(fraction_Q)

    # make_plot(alphas, fractions, save=True)
    # hyperplane = get_perpendicular_vector(w)
    # hyperplane = hyperplane * 2
    # print("weight vector", w, "hyperplane", hyperplane)
    # # my ugly plotting
    # plt.scatter(np.array(data_x)[:,0], np.array(data_x)[:,1], c=data_y)
    # origin = [0, hyperplane[0]]
    # end = [0, hyperplane[1]]
    # origin1 = [0, w[0]]
    # end1 = [0, w[1]]
    # plt.plot(origin1, end1, color='green', label='weight vector')
    # plt.plot(origin, end)
    # plt.show()


if __name__ =='__main__':
    main()
