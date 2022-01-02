from matplotlib.colors import Colormap
import numpy as np
import random
import matplotlib.pyplot as plt

# somewhat rewritten function from the internet to find perpendicular vectors
def get_perpendicular_vector(w):
    if w.shape[0] == 2:
        new_v = [w[1], -w[0]]
        return new_v       
    else:
        second_v = np.zeros(w.shape[0])
        second_v[random.randint(0, w.shape[0]-1)] = 1
        return np.cross(w, second_v)

fraction_list_x = []
fraction_list_y = []
for alpha in np.arange(0.75, 3, 0.25):
    # parameters
    N = 100
    P = int(alpha * N)
    max_epochs = 100
    w = np.zeros(N)
    num_seperable = 0
    num_not_seperable = 0
    for n in range(0, 50):
        # generate random data
        data_x = []
        data_y = []
        for i in range(P):
            feature_vector = []
            
            data_x.append(feature_vector)
            if random.random() < 0.5:
                data_y.append(-1)
                for j in range(N):
                    feature_vector.append(random.uniform(-1, 1))
            else:
                data_y.append(1)
                for j in range(N):
                    feature_vector.append(random.uniform(-1, 1))
        # convert to numpy makes using it easier
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        # train w
        for epoch in range(max_epochs):
            for example in range(P):
                E_mu = np.dot(w, (data_x[example] * data_y[example]))
                if E_mu <= 0:
                    update = 1/N * np.array(data_x[example]) * data_y[example]
                    w += update
        # get how many points are correct / false
        not_lin_seperable = False
        for example in range(P):
            label = np.dot(w, (data_x[example] * data_y[example]))
            if label > 0:
                continue
            else:
                not_lin_seperable = True
        if not_lin_seperable:
            num_not_seperable += 1
        else:
            num_seperable += 1
    fraction_Q = num_seperable / (num_seperable+num_not_seperable)
    fraction_list_x.append(alpha)
    fraction_list_y.append(fraction_Q)
print(fraction_list_x, fraction_list_y)
plt.plot(fraction_list_x, fraction_list_y)
plt.show()


# print(correct, wrong)
# # w defines a plane that it is perpendicular to
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
