from __future__ import division

import numpy as np

from copy import deepcopy
from functools import reduce
from functools import partial


class Fish(object):

    INITIAL_WEIGHT = 1.0
    INITIAL_FITNESS = np.inf
    INITIAL_FITNESS_DIFFERENCE = 0.0

    def __init__(self, obj_function):
        self.pos = obj_function.sample()
        self.weight = Fish.INITIAL_WEIGHT
        self.fitness = Fish.INITIAL_FITNESS
        self.delta_pos = np.zeros(shape=len(self.pos))
        self.delta_fitness = Fish.INITIAL_FITNESS_DIFFERENCE


class FSS(object):

    def __init__(self, obj_function, school_size=30, n_iter=5000,
                 initial_step_in=1, final_step_in=0.01,
                 initial_step_vol=1, final_step_vol=0.01):
        self.school = []
        self.school_size = school_size
        self.obj_function = obj_function

        self.n_iter = n_iter
        self.initial_step_in = initial_step_in
        self.final_step_in = final_step_in
        self.step_in = initial_step_in

        self.initial_step_vol = initial_step_vol
        self.final_step_vol = final_step_vol
        self.step_vol = initial_step_vol

    def __reset_optimizer(self):
        self.school = []
        self.step_in = self.initial_step_in
        self.step_vol = self.initial_step_vol
        self.optimality_tracking = []
        self.optimal_solution = None

    def __init_school(self):
        for _ in range(self.school_size):
            self.school.append(Fish(self.obj_function))
        self.total_weight = sum(map(lambda fish: fish.weight, self.school))

    def __evaluate_school(self):
        evaluate = partial(FSS.__evaluate_fish, obj_function=self.obj_function)
        map(evaluate, self.school)

    def __update_optimal_solution(self):
        n_optimal_solution = min(self.school, key=lambda fish: fish.fitness)
        if not self.optimal_solution:
            self.optimal_solution = deepcopy(n_optimal_solution)
        elif n_optimal_solution.fitness <= self.optimal_solution.fitness:
                self.optimal_solution = deepcopy(n_optimal_solution)

    def __update_optimality_tracking(self):
        self.optimality_tracking.append(self.optimal_solution.fitness)

    def __update_total_weight(self):
        self.total_weight = sum(map(lambda fish: fish.weight, self.school))

    def __apply_individual_movement_operator(self):
        individual_movement_operator = partial(FSS.__individual_movement_operator,
                                               obj_function=self.obj_function,
                                               step=self.step_in)
        map(individual_movement_operator, self.school)

    def __apply_feeding_operator(self):
        max_delta_fitness = max(self.school, key=lambda fish: fish.delta_fitness).delta_fitness
        feeding_operator = partial(FSS.__feeding_operator, max_delta_fitness=max_delta_fitness)
        map(feeding_operator, self.school)

    def __apply_collective_instinctive_movement_operator(self):
        sum_delta_fitness = sum(map(lambda fish: fish.delta_fitness, self.school))
        instinctive_movement = map(lambda fish: fish.delta_fitness * fish.delta_pos, self.school)
        instinctive_movement = reduce(lambda x, y: x + y, instinctive_movement)/sum_delta_fitness
        collective_instinctive = partial(FSS.__collective_instinctive_movement_operator,
                                         instinctive_movement=instinctive_movement)
        map(collective_instinctive, self.school)

    def __calculate_barycenter(self):
        sum_weights = sum(map(lambda fish: fish.weight, self.school))
        barycenter = map(lambda fish: fish.pos * fish.weight, self.school)
        self.barycenter = reduce(lambda x, y: x + y, barycenter) / sum_weights

    def __apply_collective_volitive_movement(self):
        collective_volitive_movement = partial(FSS._collective_volitive_movement,
                                               barycenter=self.barycenter, total_weight=self.total_weight,
                                               step_vol=self.step_vol, fish_school=self.school)
        map(collective_volitive_movement, self.school)

    def __update_step(self, itr):
        if self.step_in > self.final_step_in:
            self.step_in -= (self.final_step_in - self.initial_step_in)/(itr + 1)

        if self.step_vol > self.final_step_vol:
            self.step_vol -= (self.final_step_vol - self.initial_step_vol) / (itr + 1)

    def optimize(self):
        self.__reset_optimizer()
        self.__init_school()
        for itr in range(self.n_iter):
            self.__evaluate_school()
            self.__update_optimal_solution()

            self.__apply_individual_movement_operator()
            self.__apply_feeding_operator()

            self.__evaluate_school()
            self.__update_optimal_solution()
            self.__update_optimality_tracking()

            self.__apply_collective_instinctive_movement_operator()
            self.__calculate_barycenter()
            self.__apply_collective_volitive_movement()

            self.__update_total_weight()
            self.__update_step(itr)
            print("iter: {} = cost: {}".format(itr, "%04.03e" % self.optimal_solution.fitness))

    @staticmethod
    def __evaluate_fish(fish, obj_function):
        fish.fitness = obj_function.evaluate(fish.pos)

    @staticmethod
    def __evaluate_boundaries(pos, obj_function):
        if (pos < obj_function.minf).any() or (pos > obj_function.maxf).any():
            pos[pos > obj_function.maxf] = obj_function.maxf
            pos[pos < obj_function.minf] = obj_function.minf
        return pos

    @staticmethod
    def __individual_movement_operator(fish, obj_function, step):
        n_candidate_pos = np.random.uniform(low=-1, high=1, size=obj_function.dim) * step + fish.pos
        n_candidate_pos = FSS.__evaluate_boundaries(n_candidate_pos, obj_function)
        n_fitness_evaluation = obj_function.evaluate(n_candidate_pos)
        fish.delta_fitness = n_fitness_evaluation - fish.fitness
        if n_fitness_evaluation < fish.fitness:
            fish.pos = n_candidate_pos
            fish.fitness = n_fitness_evaluation

    @staticmethod
    def __feeding_operator(fish, max_delta_fitness):
        fish.weight += fish.delta_fitness/max_delta_fitness

    @staticmethod
    def __collective_instinctive_movement_operator(fish, instinctive_movement):
        fish.pos += instinctive_movement

    @staticmethod
    def _collective_volitive_movement(fish, barycenter, total_weight, step_vol, fish_school):
        current_total_weight = sum(map(lambda fish: fish.weight, fish_school))
        off_set = step_vol * np.random.uniform(low=0.1, high=0.9) *\
                  (fish.pos - barycenter) / np.linalg.norm(fish.pos - barycenter)
        if current_total_weight > total_weight:
            fish.pos -= off_set
        else:
            fish.pos += off_set
