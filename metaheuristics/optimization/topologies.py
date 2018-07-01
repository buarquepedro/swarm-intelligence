import numpy as np

from abc import ABCMeta
from abc import abstractmethod
from six import add_metaclass


@add_metaclass(ABCMeta)
class Topology(object):

    def __init__(self, c1=2.05, c2=2.05, v_max=1000):
        self.c1 = c1
        self.c2 = c2

        self.swarm = []
        self.v_max = v_max

        self.dim = None
        self.minf = None
        self.maxf = None

        self.obj_function = None
        self.optimal_particle = None

    def load(self, obj_function, swarm, c1, c2, v_max):
        self.c1 = c1
        self.c2 = c2

        self.swarm = swarm
        self.v_max = v_max

        self.obj_function = obj_function
        self.dim = self.obj_function.dim
        self.minf = self.obj_function.minf
        self.maxf = self.obj_function.maxf

    def evaluate_particle_boundaries(self, p):
        if (p.pos < self.minf).any() or (p.pos > self.maxf).any():
            p.speed[p.pos < self.minf] = -1 * p.speed[p.pos < self.minf]
            p.speed[p.pos > self.maxf] = -1 * p.speed[p.pos > self.maxf]
            p.pos[p.pos > self.maxf] = self.maxf
            p.pos[p.pos < self.minf] = self.minf

    @abstractmethod
    def extract_optimal_candidate(self):
        raise NotImplementedError

    @abstractmethod
    def get_social_component(self, p):
        raise NotImplementedError

    @abstractmethod
    def update_particles_components(self, w, inertia_update_mode='decreasing'):
        raise NotImplementedError


class GlobalTopology(Topology):

    def __init__(self, *args, **kwargs):
        super(GlobalTopology, self).__init__(*args, **kwargs)
        self.name = 'Global Topology'

    def get_social_component(self, p):
        return self.extract_optimal_candidate().pbest_pos

    def update_particle_component(self, p, w, inertia_update_mode):
        social_component = self.get_social_component(p)
        r1 = np.random.random(len(p.speed))
        r2 = np.random.random(len(p.speed))
        if inertia_update_mode == 'clerc':
            p.speed = w * (np.array(p.speed) + self.c1 * r1 * (p.pbest_pos - p.pos) + self.c1 * r2 * (
                    social_component - p.pos))
        else:
            p.speed = w * np.array(p.speed) + self.c1 * r1 * (p.pbest_pos - p.pos) + \
                      self.c1 * r2 * (social_component - p.pos)
        p.speed = np.sign(p.speed) * np.minimum(np.absolute(p.speed), np.ones(self.dim) * self.v_max)
        p.pos = p.pos + p.speed

    def extract_optimal_candidate(self):
        return min(self.swarm, key=lambda particle: particle.pbest_cost)

    def update_particles_components(self, w, inertia_update_mode='decreasing'):
        for p in self.swarm:
            self.update_particle_component(p, w, inertia_update_mode)
            self.evaluate_particle_boundaries(p)


class FocalTopology(GlobalTopology):

    def __init__(self, *args, **kwargs):
        super(FocalTopology, self).__init__(*args, **kwargs)
        self.focal_best = None
        self.is_focal_selected = False
        self.name = 'Focal Topology'

    def get_social_component(self, p):
        social_component = self.extract_optimal_candidate().pbest_pos
        if p == self.focal_best:
            social_component = min(self.swarm, key=lambda p: p.pbest_cost).pbest_pos
        return social_component

    def update_particles_components(self, w, inertia_update_mode='decreasing'):
        if not self.is_focal_selected:
            self.focal_best = self.__define_focal()
            self.is_focal_selected = True
        super(FocalTopology, self).update_particles_components(w, inertia_update_mode)

    def __define_focal(self):
        return np.random.choice(self.swarm)

    def extract_optimal_candidate(self):
        return self.focal_best


class LocalTopology(GlobalTopology):

    def __init__(self, n_neighbors=2, *args, **kwargs):
        super(LocalTopology, self).__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors
        self.name = 'Local Topology'

    def get_social_component(self, p):
        local_best = self.__get_local_best(p)
        return local_best.pbest_pos

    def __get_local_best(self, particle):
        neighbors = self.__get_neighbors(particle)
        return min(neighbors, key=lambda p: p.pbest_cost)

    def __get_neighbors(self, particle):
        swarm = np.array(self.swarm)
        dist = map(lambda p: np.linalg.norm(p.pos - particle.pos), self.swarm)
        sorted_dist = np.argsort(dist)
        return swarm[sorted_dist[1:self.n_neighbors]]

