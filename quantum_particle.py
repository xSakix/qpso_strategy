import numpy as np
from sklearn.preprocessing import StandardScaler


class QuantumParticle:
    def __init__(self, m=2):
        self.M = m
        self.w = np.random.dirichlet(np.ones(m), size=1)[0]
        self.p_w = np.array(self.w)
        self.g_w = np.array(self.w)
        self.c = np.array(self.w)
        self.alpha = 0.75
        self.last_fitness = None
        self.fitness = None

    def compute_weights(self):
        phi = np.random.uniform(0., 1.)
        p = np.add(np.multiply(phi, self.p_w), np.multiply(np.subtract(1., phi), self.g_w))
        u = np.random.uniform(0., 1.)
        for i in range(len(self.w)):
            if np.random.uniform(0., 1.) < 0.5:
                self.w[i] = p[i] + self.alpha * np.abs(self.w[i] - self.c[i]) * np.log(1. / u)
            else:
                self.w[i] = p[i] - self.alpha * np.abs(self.w[i] - self.c[i]) * np.log(1. / u)

            if self.w[i] < 0.:
                self.w[i] = 0.

        # normalize
        self.w = np.divide(self.w, np.sum(self.w))
        self.w = np.round(self.w, 2)

    def evaluate(self, data):
        return np.array(np.cumsum(np.sum(self.w * data)))

    def compute_fitness(self, data):
        y = self.evaluate(data)
        self.fitness = y[-1]

        if self.last_fitness is None or self.last_fitness < self.fitness:
            self.last_fitness = self.fitness
            self.p_w = self.w
