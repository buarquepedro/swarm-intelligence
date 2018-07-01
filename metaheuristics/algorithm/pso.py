from __future__ import division

import numpy as np

from copy import deepcopy
from functools import partial


class Particle(object):

    def __init__(self, obj_function):
        self.cost = np.inf
        self.pos = obj_function.sample()
        self.speed = np.zeros(len(self.pos))
        self.pbest_pos = self.pos
        self.pbest_cost = self.cost


class PSO(object):

    def __init__(self, obj_function, topology, swarm_size=50, n_iter=1000, lb_w=0.4, up_w=0.8,
                 c1=2.05, c2=2.05, v_max=100000, inertia_update_mode='decreasing'):
        self.obj_function = obj_function
        self.topology = topology
        self.swarm_size = swarm_size
        self.n_iter = n_iter

        self.up_w = up_w
        self.lb_w = lb_w

        self.w = up_w
        self.c1 = c1
        self.c2 = c2
        self.v_max = min(v_max, 100000)
        self.inertia_update_mode = inertia_update_mode

    def __reset_optimizer(self):
        self.swarm = []
        self.w = self.up_w
        self.optimality_tracking = []
        self.topology.load(self.obj_function, self.swarm, self.c1, self.c2, self.v_max)

    def __init_swarm(self):
        for _ in range(self.swarm_size):
            self.swarm.append(Particle(self.obj_function))
        self.__evaluate_particles()
        self.optimal_particle = self.__extract_optimal_particle()

    def __evaluate_particles(self):
        evaluate = partial(PSO.__evaluate_particle, obj_function=self.obj_function)
        map(evaluate, self.swarm)

    def __update_inertia(self, itr):
        if self.inertia_update_mode == 'decreasing':
            self.w = self.up_w - (float(itr) / self.n_iter) * (self.up_w - self.lb_w)
        elif self.inertia_update_mode == 'constant':
            self.w = self.up_w
        elif self.inertia_update_mode == 'clerc':
            phi = self.c1 + self.c2
            self.w = 2. / np.abs(2 - phi - np.sqrt(phi**2 - 4*phi))

    def __update_optimal_solution(self):
        n_optimal_solution = self.topology.extract_optimal_candidate()
        if not self.optimal_particle:
            self.optimal_particle = deepcopy(n_optimal_solution)
        else:
            if n_optimal_solution.pbest_cost <= self.optimal_particle.pbest_cost:
                self.optimal_particle = deepcopy(n_optimal_solution)
        self.optimality_tracking.append(self.optimal_particle.pbest_cost)

    def __extract_optimal_particle(self):
        return self.topology.extract_optimal_candidate()

    def optimize(self):
        self.__reset_optimizer()
        self.__init_swarm()
        for itr in range(self.n_iter):
            self.topology.update_particles_components(self.w, self.inertia_update_mode)
            self.__evaluate_particles()
            self.__update_optimal_solution()
            self.__update_inertia(itr)
            print("iter: {} = cost: {}".format(itr, "%04.03e" % self.optimal_particle.cost))

    @staticmethod
    def __evaluate_particle(particle, obj_function):
        particle.cost = obj_function.evaluate(particle.pos)
        if particle.cost < particle.pbest_cost:
            particle.pbest_pos = np.array(particle.pos)
            particle.pbest_cost = particle.cost
