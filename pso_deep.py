from metaheuristics.algorithm.pso import PSO

from metaheuristics.optimization.topologies import LocalTopology
from metaheuristics.optimization.topologies import GlobalTopology
from metaheuristics.optimization.topologies import FocalTopology

from metaheuristics.optimization.objection_function import DeepNeuralNetworkObjectiveFunction


def get_topology(topology, n_neighbors=5):
    topologies = {'Global': GlobalTopology(), 'Focal': FocalTopology(), 'Local': LocalTopology(n_neighbors=n_neighbors)}
    return topologies[topology]


def main():
    object_function = DeepNeuralNetworkObjectiveFunction(3)
    optimizer = PSO(obj_function=object_function, topology=get_topology('Local'),
                    swarm_size=30, n_iter=100, v_max=100.0, inertia_update_mode='decreasing')
    optimizer.optimize()


if __name__ == '__main__':
    main()
