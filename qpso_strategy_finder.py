from quantum_particle import QuantumParticle
import numpy as np


class QPSOStrategyFinder:
    def __init__(self, population_size, iterations, m=2):
        self.M = m
        self.population_size = population_size
        self.iterations = iterations
        self.population = []
        for i in range(self.population_size):
            self.population.append(QuantumParticle(self.M))

    def run(self, data):

        for i in range(self.iterations):
            sum_of_weights = np.zeros(self.M)
            for p in self.population:
                sum_of_weights = np.add(sum_of_weights, p.p_w)
            c = np.divide(sum_of_weights, float(self.population_size))
            for p in self.population:
                p.c = c
                p.compute_weights()
                p.compute_fitness(data)

            self.population.sort(key=lambda particle: particle.fitness)

            for p in self.population:
                p.g_w = self.population[-1].w

            # print('iteration(%d) = %f | %s' % (i, self.population[-1].fitness, str(self.population[-1].w)))
            print('iteration(%d) = %f ' % (i, self.population[-1].fitness))

        return self.population[-1]
