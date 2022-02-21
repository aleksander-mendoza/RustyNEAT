import numpy as np
from matplotlib import pyplot as plt

n = 6
m = 8
nn = 2 ** n
P = np.random.rand(nn)
P = P / P.sum()
W = np.random.rand(n, m)
W = W / W.sum(0)
q_given_j = np.zeros(nn)
p_of_y = np.empty(m)


def x_as_vec(x):
    return [x >> i & 1 for i in range(n - 1, -1, -1)]


X = np.array([x_as_vec(x_num) for x_num in range(2 ** n)])
lr = 1
s = []
d = []
l1 = []
l2 = []
kld = []
E = np.empty((n, m))
fig, axs = plt.subplots(2, 2)
fig.suptitle("learning_rate=" + str(lr) + ", n="+str(n)+", m="+str(m))

while True:
    p_of_y.fill(0)
    E.fill(0)
    S = X @ W
    K = S.argmax(1)
    Sk = np.take_along_axis(S, np.expand_dims(K, 1), 1).squeeze()
    E_s_k = (Sk*P).sum()
    Px = (W[:, K].T - (X - 1))
    Q_given_y = Px.prod(1)
    for x_num, (p, x, k) in enumerate(zip(P, X, K)):
        E[:, k] += x * p
        p_of_y[k] += p
    P_given_y = P / p_of_y[K]
    # l2_dist = np.sum(np.square(P_given_y - Q_given_y))
    l1_dist = np.sum(np.abs(P_given_y - Q_given_y))
    l = np.log(P_given_y/Q_given_y)
    l[Q_given_y==0] = 0
    kl_div = np.sum(P_given_y*l)

    E_sum = E.sum(0)
    E_sum[E_sum == 0] = 1  # prevent division by 0
    Ej = E / E_sum
    w_E_diff = np.sum(np.abs(W - Ej))
    W = (1 - lr) * W + lr * Ej

    d.append(w_E_diff)
    s.append(E_s_k)
    l1.append(l1_dist)
    # l2.append(np.sqrt(l2_dist))
    kld.append(kl_div)
    if len(s) % 300==0:
        axs[0, 0].cla()
        axs[1, 0].cla()
        axs[0, 1].cla()
        axs[1, 1].cla()
        axs[0, 0].set_title(r"$\mathbb{E}(s_k)$")
        axs[1, 0].set_title(r"$||cW_j-\mathbb{E}({\bf x}|y_j)||_2$")
        axs[0, 1].set_title(r"$\sum_j||p({\bf x}|y_j)-q({\bf x}|y_j)||_1$")
        axs[1, 1].set_title(r"$D_{KL}(p||q)$")
        axs[0, 0].plot(s)
        axs[1, 0].plot(d)
        axs[0, 1].plot(l1)
        axs[1, 1].plot(kld)
        print(E)
    plt.pause(0.001)
