import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metaheuristics.algorithm.pso import PSO

from metaheuristics.optimization.topologies import LocalTopology
from metaheuristics.optimization.topologies import GlobalTopology
from metaheuristics.optimization.topologies import FocalTopology

from metaheuristics.optimization.objection_function import Rastrigin
from metaheuristics.optimization.objection_function import Rosenbrock
from metaheuristics.optimization.objection_function import Sphere
from metaheuristics.optimization.objection_function import Schwefel


df = pd.DataFrame()


def get_objective(objective, dimension=30):
    objectives = {'Sphere': Sphere(dimension),
                  'Rastrigin': Rastrigin(dimension),
                  'Rosenbrock': Rosenbrock(dimension),
                  'Schwefel': Schwefel(dimension)}
    return objectives[objective]


def get_topology(topology, n_neighbors=5):
    topologies = {'Global': GlobalTopology(), 'Focal': FocalTopology(), 'Local': LocalTopology(n_neighbors=n_neighbors)}
    return topologies[topology]


def simulate(obj_function, swarm_topology, swarm_size=30, n_iter=10000, v_max=100.0,
             inertia_update_mode='decreasing', simulations=30):
    itr = range(n_iter)
    values = np.zeros(n_iter)
    box_optimal = []
    for _ in range(simulations):
        optimizer = PSO(obj_function=get_objective(obj_function), topology=get_topology(swarm_topology),
                        swarm_size=swarm_size, n_iter=n_iter, v_max=v_max,
                        inertia_update_mode=inertia_update_mode)
        optimizer.optimize()
        values += np.array(optimizer.optimality_tracking)
        box_optimal.append(optimizer.optimal_particle.pbest_cost)
        print(optimizer.optimal_particle.pbest_pos)
    values /= simulations

    title = obj_function + ' Function, inertia weight ' + inertia_update_mode
    df[swarm_topology] = box_optimal
    plt.plot(itr, values, lw=0.5, label=swarm_topology)
    plt.legend(loc='upper right')
    plt.title(title)


def main():
    global df

    # LINEAR DECREASING ------------------------------------------------------------------------------------
    OBJECTIVE_FUNCTION = 'Rosenbrock'
    plt.figure(figsize=(12, 7))
    df = pd.DataFrame()
    simulate(OBJECTIVE_FUNCTION, 'Global', swarm_size=30, n_iter=10000,
             v_max=0.1, inertia_update_mode='decreasing')
    simulate(OBJECTIVE_FUNCTION, 'Focal', swarm_size=30, n_iter=10000,
             v_max=0.1, inertia_update_mode='decreasing')
    simulate(OBJECTIVE_FUNCTION, 'Local', swarm_size=30, n_iter=10000,
             v_max=0.1, inertia_update_mode='decreasing')
    plt.savefig('figs/rosenbrock_decreasing.jpg')
    df.to_csv('rosenbrock_decreasing.csv')

    # CONSTANT ----------------------------------------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    df = pd.DataFrame()
    simulate(OBJECTIVE_FUNCTION, 'Global', swarm_size=30, n_iter=10000,
             v_max=0.1, inertia_update_mode='constant')
    simulate(OBJECTIVE_FUNCTION, 'Focal', swarm_size=30, n_iter=10000,
             v_max=0.1, inertia_update_mode='constant')
    simulate(OBJECTIVE_FUNCTION, 'Local', swarm_size=30, n_iter=10000,
             v_max=0.1, inertia_update_mode='constant')
    plt.savefig('figs/rosenbrock_constant.jpg')
    df.to_csv('rosenbrock_constant.csv')

    # CLERC --------------------------------------------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    df = pd.DataFrame()
    simulate(OBJECTIVE_FUNCTION, 'Global', swarm_size=30, n_iter=10000,
             v_max=0.1, inertia_update_mode='clerc')
    simulate(OBJECTIVE_FUNCTION, 'Focal', swarm_size=30, n_iter=10000,
             v_max=0.1, inertia_update_mode='clerc')
    simulate(OBJECTIVE_FUNCTION, 'Local', swarm_size=30, n_iter=10000,
             v_max=0.1, inertia_update_mode='clerc')
    plt.savefig('figs/rosenbrock_clerc.jpg')
    df.to_csv('rosenbrock_clerc.csv')

    plt.show()


if __name__ == '__main__':
    main()
