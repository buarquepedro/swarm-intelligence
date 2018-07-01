from __future__ import division

import random
import numpy as np

from functools import partial


class Genome(object):

    BINARY_BASE = 2

    def __init__(self, dim):
        self.gene = Genome.generate_random_genome(dim)
        self.cross_op = None
        self.mutation_op = None
        self.cost = -1

    def mutate(self, prob):
        if random.random() <= prob:
            if not self.mutation_op:
                idx = np.random.choice(range(len(self.gene)))
                self.gene[idx] = 1 - self.gene[idx]
            else:
                self.gene = self.mutation_op(self.gene)

    def crossover(self, ga, gb):
        if not self.cross_op:
            idx = np.random.choice(range(len(ga)))
            self.gene[:idx] = ga[:idx]
            self.gene[idx:] = gb[idx:]
        else:
            self.gene = self.cross_op(ga, gb)

    @staticmethod
    def generate_random_genome(dim):
        return np.random.randint(Genome.BINARY_BASE, size=dim)


class BinaryGA(object):

    def __init__(self, fitness, pop_size=1000, mutation_rate=0.9, cross_rate=1.0,
                 max_iter=5000, maximize=True, elitism=True):
        self.dim = fitness.dim
        self.fitness = fitness
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.cross_rate = cross_rate
        self.max_iter = max_iter
        self.elitism = elitism
        self.cross_op = None
        self.mutation_op = None
        self.op = max if maximize else min
        self.nop = min if maximize else max

        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []
        self.optimal_solution = None

    def __initialize_population(self):
        self.genes = []
        for _ in xrange(self.pop_size):
            individual = Genome(dim=self.dim)
            self.genes.append(individual)

        if self.cross_op:
            operator = partial(BinaryGA.__set_cross_op, self.cross_op)
            map(operator, self.genes)

        if self.mutation_op:
            operator = partial(BinaryGA.__set_mutation_op, self.mutation_op)
            map(operator, self.genes)

        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []
        self.optimal_solution = np.random.choice(self.genes)

    def optimize(self):
        self.__initialize_population()
        self.__evaluate_population()

        for itr in xrange(self.max_iter):
            # print('LOG : Iteration {} --- GBEST: {}'.format(itr, self.optimal_solution.cost))
            self.__get_fittest()
            self.__induce_cross_over()
            self.__induce_mutation()
            self.__retrieve()
            self.__evaluate_population()
            self.optimum_cost_tracking_iter.append(self.optimal_solution.cost)

    def best(self):
        return self.optimal_solution.gene

    def set_cross_operator(self, op):
        self.cross_op = op

    def set_mutation_operator(self, op):
        self.mutation_op = op

    def __evaluate_population(self):
        evaluate = partial(BinaryGA.__evaluate, self.fitness)
        map(evaluate, self.genes)
        self.optimal_solution = self.op(self.genes, key=lambda g: g.cost)

    def __get_fittest(self):
        if self.elitism:
            self.genes.remove(self.optimal_solution)

    def __retrieve(self):
        if self.optimal_solution not in self.genes:
            self.genes.append(self.optimal_solution)

    def __induce_mutation(self):
        mutation = partial(BinaryGA.__mutation, self.mutation_rate)
        map(mutation, self.genes)

    def __induce_cross_over(self):
        if random.random() <= self.cross_rate:
            inferior_gene = self.nop(self.genes, key=lambda g: g.cost)
            self.genes.remove(inferior_gene)
            idx1, idx2 = self.__pick_parents()
            inferior_gene.crossover(self.genes[idx1].gene, self.genes[idx2].gene)
            self.genes.append(inferior_gene)

    def __pick_parents(self):
        idx1 = BinaryGA.__roulette_selection(self.genes)
        idx2 = BinaryGA.__roulette_selection(self.genes)
        while not idx2 != idx1:
            idx2 = BinaryGA.__roulette_selection(self.genes)
        return idx1, idx2

    @staticmethod
    def __set_cross_op(op, genome):
        genome.cross_op = op

    @staticmethod
    def __set_mutation_op(op, genome):
        genome.mutation_op = op

    @staticmethod
    def __evaluate(fitness, genome):
        genome.cost = fitness.evaluate(genome.gene)

    @staticmethod
    def __mutation(prob, genome):
        genome.mutate(prob)

    @staticmethod
    def __roulette_selection(genes):
        # TODO study if its possible to optimize this bunch of code
        sorted_genes = sorted(genes, key=lambda genome: genome.cost)
        indices, sorted_weights = zip(*enumerate(sorted_genes))
        all_costs = map(lambda g_a: g_a.cost, sorted_genes)
        total_sum = reduce(lambda g_a, g_b: g_a + g_b, all_costs)
        probs = map(lambda g_a: g_a.cost / total_sum, genes)
        cum_prob = np.cumsum(probs)

        # TODO find if its possible to get rid of this loop
        random_num = random.random()
        for index_value, cum_prob_value in zip(indices, cum_prob):
            if random_num < cum_prob_value:
                return index_value
