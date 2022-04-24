import numpy as np


def quadratic_expansion(x):
    assert len(x.shape) == 2
    batch = x.shape[0]
    x_ = np.expand_dims(x, 1)
    xx = x_.transpose(0, 2, 1) @ x_
    xx = xx.reshape(batch, -1)
    x2 = np.hstack([xx, x])
    return x2


def discrete_derivative(x):
    der = x[1:] - x[:-1]
    return der


class DeMean:
    def __init__(self, x):
        self.mu = x.mean(0)

    def __call__(self, x):
        x = x - self.mu
        return x


class IncDeMean:
    def __init__(self, dim):
        self.dim = dim
        self.mu = np.zeros(dim)
        self.n = 0

    def __call__(self, x):
        assert x.shape == (1, self.dim)
        self.mu = (self.n * self.mu + x.sum(axis=0)) / (self.n + x.shape[0])
        self.n += x.shape[0]
        x = x - self.mu
        return x


class IncDeMeanMovAvg:
    def __init__(self, dim, epsilon):
        self.dim = dim
        self.mu = np.zeros(dim)
        self.epsilon = epsilon

    def __call__(self, x):
        assert x.shape == (1, self.dim)
        for i in x:
            self.mu = (1 - self.epsilon) * self.mu + self.epsilon * i
        x = x - self.mu
        return x


class IncDiscreteDerivative:
    def __init__(self, dim):
        self.dim = dim
        self.prev = np.zeros(dim)

    def __call__(self, x):
        assert x.shape == (1, self.dim)
        d = x - self.prev
        self.prev = x.copy()
        return d


class IncEpsilonForMean:
    def __init__(self):
        self.n = 0

    def __call__(self, x):
        self.n += x.shape[0]
        return 1. / self.n


class IncEpsilonForMovAvg:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, x):
        return self.epsilon


class IncEpsilonAmnesic:
    def __init__(self, n1=20, n2=200, m=2000, C=3):
        self.n1 = n1
        self.n2 = n2
        self.m = m
        self.C = C
        self.i = 0

    def __call__(self, x):
        self.i += x.shape[0]
        if self.i < self.n1:
            l = 0
        elif self.i < self.n2:
            l = self.C * (self.i - self.n1) / (self.n2 - self.n1)
        else:
            l = self.C + (self.i - self.n2) / self.m
        _wnew = float(1 + l) / self.i
        return [1 - _wnew, _wnew]


class IncPca:

    def __init__(self, de_mean):
        self.de_mean = de_mean

    def __call__(self, x):
        assert (x.shape[0] == 1)
        self.n += 1

        x = self.de_mean(x)

        red_j = self.output_dim
        red_j_Flag = False
        explained_var = 0.0

        r = x.copy()
        for j in range(self.output_dim):
            v = self._v[j:j + 1].copy()
            v = w1 * v + w2 * np.dot(r, v.T) / self._d[j] * r
            self._d[j] = np.linalg.norm(v)
            vn = v / self._d[j]
            r = r - np.dot(r, vn.T) * vn
            explained_var += self._d[j]
            if (self.reduce is True) and (red_j_Flag is False):
                ratio1 = self._d[j] / self._d[0]
                ratio2 = explained_var / self.explained_var_tot
                # print j, " :  ", ratio1, " :  ", ratio2, " :  ",self._d[j]
                if ratio1 < self.var_rel or ratio2 > self.beta:
                    red_j = j
                    red_j_Flag = True
                    # print j,  " :  ", ratio1, " :  ", ratio2, " :  ", self._d[j]
            self._v[j] = v.copy()
            self._vn[j] = vn.copy()

        if explained_var > 0.0001:
            self.explained_var_tot = explained_var
        self.v = self._vn[:red_j].copy()
        self.d = self._d[:red_j].copy()
        self.reducedDim = red_j


class NormaliseStd:

    def __init__(self, x):
        self.std = x.std(0)

    def __call__(self, x):
        x = x / self.std
        return x


def cov(x):
    batch = x.shape[0]
    x_cov = x.T @ x / (batch - 1)
    return x_cov


def sort_eigenvalues(e_values, e_vectors):
    e_inc_order = np.flip(e_values.argsort())
    e_values = e_values[e_inc_order]
    e_vectors = e_vectors[:, e_inc_order]  # note that we have to re-order the columns, not rows
    return e_values, e_vectors


class TruncateEigenvalues:
    def __init__(self, cut_off=1e-8):
        self.cut_off = cut_off

    def __call__(self, e_values, e_vectors):
        i = e_values > self.cut_off
        vec = e_vectors[:, i]
        val = e_values[i]
        val, vec = sort_eigenvalues(val, vec)
        return val, vec


class RetainKEigenvalues:
    def __init__(self, k):
        self.k = k

    def __call__(self, e_values, e_vectors):
        e_values, e_vectors = sort_eigenvalues(e_values, e_vectors)
        vec = e_vectors[:, :self.k]
        val = e_values[:self.k]
        return val, vec


def no_truncation(e_values, e_vectors):
    return e_values, e_vectors


def pca(x):
    x_cov = cov(x)
    e_values, e_vectors = np.linalg.eigh(x_cov)
    return e_values, e_vectors


def whitening_transformation(x, truncation):
    e_values, e_vectors = pca(x)
    e_values, e_vectors = truncation(e_values, e_vectors)
    whitening = e_vectors / np.sqrt(e_values)
    return whitening


class PcaWhitening:
    def __init__(self, x, truncation):
        self.de_mean = DeMean(x)
        x = self.de_mean(x)
        self.whitening = whitening_transformation(x, truncation)

    def __call__(self, x):
        x = self.de_mean(x)
        x = x @ self.whitening
        return x


class Sfa:

    def __init__(self, x, truncation):
        self.whitening = PcaWhitening(x, truncation)
        x = self.whitening(x)
        x_deriv = discrete_derivative(x)
        e_values, e_vectors = pca(x_deriv)
        self.e_vectors = e_vectors

    def __call__(self, x):
        x = self.whitening(x)
        x = x @ self.e_vectors
        return x


class Sfa2:
    def __init__(self, x, truncation):
        x = quadratic_expansion(x)
        self.sfa = Sfa(x, truncation)

    def __call__(self, x):
        x = quadratic_expansion(x)
        return self.sfa(x)
