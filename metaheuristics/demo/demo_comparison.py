import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metaheuristics.algorithm.fss import FSS
from metaheuristics.algorithm.pso import PSO
from metaheuristics.algorithm.abc import ABC

from metaheuristics.optimization.topologies import LocalTopology
from metaheuristics.optimization.topologies import GlobalTopology
from metaheuristics.optimization.topologies import FocalTopology

from matplotlib.style import use

from metaheuristics.optimization.objection_function import Rastrigin
from metaheuristics.optimization.objection_function import Rosenbrock
from metaheuristics.optimization.objection_function import Sphere
from metaheuristics.optimization.objection_function import Schwefel

use('classic')
df = pd.DataFrame()


def get_objective(objective, dimension=30):
    objectives = {'Sphere': Sphere(dimension),
                  'Rastrigin': Rastrigin(dimension),
                  'Rosenbrock': Rosenbrock(dimension),
                  'Schwefel': Schwefel(dimension)}
    return objectives[objective]


def get_topology(topology, n_neighbors=5):
    topologies = {'Global': GlobalTopology(),
                  'Focal': FocalTopology(),
                  'Local': LocalTopology(n_neighbors=n_neighbors)}
    return topologies[topology]


def simulate_pso(obj_function, swarm_topology, swarm_size=30, n_iter=5000, v_max=100.0,
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

    df['pso_' + obj_function] = box_optimal
    plt.plot(itr, values, lw=0.5, label='PSO')


def simulate_fss(obj_function, school_size=30, n_iter=5000, initial_step_in=0.1, final_step_in=0.001,
                 initial_step_vol=0.01, final_step_vol=0.001, simulations=30):
    itr = range(n_iter)
    values = np.zeros(n_iter)
    box_optimal = []
    for _ in range(simulations):
        optimizer = FSS(obj_function=get_objective(obj_function),
                        school_size=school_size, n_iter=n_iter,
                        initial_step_in=initial_step_in, final_step_in=final_step_in,
                        initial_step_vol=initial_step_vol, final_step_vol=final_step_vol)
        optimizer.optimize()
        values += np.array(optimizer.optimality_tracking)
        box_optimal.append(optimizer.optimal_solution.fitness)
        print(optimizer.optimal_solution.pos)
    values /= simulations

    df['fss_' + obj_function] = box_optimal
    plt.plot(itr, values, lw=0.5, label='FSS')
    plt.xticks(rotation=45)


def simulate_abc(obj_function, colony_size=30, n_iter=5000, max_trials=100, simulations=30):
    itr = range(n_iter)
    values = np.zeros(n_iter)
    box_optimal = []
    for _ in range(simulations):
        optimizer = ABC(obj_function=get_objective(obj_function), colony_size=colony_size, n_iter=n_iter,
                        max_trials=max_trials)
        optimizer.optimize()
        values += np.array(optimizer.optimality_tracking)
        box_optimal.append(optimizer.optimal_solution.fitness)
        print(optimizer.optimal_solution.pos)
    values /= simulations

    df['abc_' + obj_function] = box_optimal
    plt.plot(itr, values, lw=0.5, label='ABC')


def main():
    global df

    N_ITER = 5000
    SIMULATIONS = 30
    OBJECTIVE_FUNCTION = 'Rastrigin'

    plt.figure(figsize=(10, 7))
    df = pd.DataFrame()

    simulate_pso(obj_function=OBJECTIVE_FUNCTION, swarm_topology='Global', swarm_size=30, n_iter=N_ITER, v_max=0.1,
                 inertia_update_mode='decreasing', simulations=SIMULATIONS)

    simulate_fss(obj_function=OBJECTIVE_FUNCTION, school_size=30, n_iter=N_ITER,
                 initial_step_in=0.01, final_step_in=0.001, initial_step_vol=0.02, final_step_vol=0.002,
                 simulations=SIMULATIONS)

    # FFS - Sphere: initial_step_in=1, final_step_in=0.01, initial_step_vol=0.01, final_step_vol=0.001
    # FSS - Rastrigin: initial_step_in=0.01, final_step_in=0.001, initial_step_vol=0.02, final_step_vol=0.002
    # FSS - Rosenbrock: initial_step_in=0.1, final_step_in=0.01, initial_step_vol=0.1, final_step_vol=0.001
    # FSS - Schwefel: initial_step_in=0.1, final_step_in=0.001, initial_step_vol=0.01, final_step_vol=0.001

    simulate_abc(obj_function=OBJECTIVE_FUNCTION, colony_size=30, n_iter=N_ITER, max_trials=100,
                 simulations=SIMULATIONS)

    plt.legend(loc='upper right')
    plt.title(OBJECTIVE_FUNCTION + ' Function')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    plt.savefig('figs/comparison_' + OBJECTIVE_FUNCTION + '.jpg')
    df.to_csv('comparison_' + OBJECTIVE_FUNCTION + '.csv')
    plt.show()


if __name__ == '__main__':
    main()
