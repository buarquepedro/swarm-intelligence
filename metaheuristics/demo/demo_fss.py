import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metaheuristics.algorithm.fss import FSS

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


def simulate(obj_function, school_size=30, n_iter=5000, initial_step_in=0.1, final_step_in=0.001,
             initial_step_vol=0.01, final_step_vol=0.001, simulations=1):
    itr = range(n_iter)
    values = np.zeros(n_iter)
    box_optimal = []
    for _ in range(simulations):
        optimizer = FSS(obj_function=get_objective(obj_function), school_size=school_size, n_iter=n_iter,
                        initial_step_in=initial_step_in, final_step_in=final_step_in,
                        initial_step_vol=initial_step_vol, final_step_vol=final_step_vol)
        optimizer.optimize()
        values += np.array(optimizer.optimality_tracking)
        box_optimal.append(optimizer.optimal_solution.fitness)
        print(optimizer.optimal_solution.pos)
    values /= simulations

    title = 'FSS - ' + obj_function + ' Function'
    df[obj_function] = box_optimal
    plt.plot(itr, values, lw=0.5, label=obj_function)
    plt.legend(loc='upper right')
    plt.title(title)


def main():
    global df

    # SPHERE FUNCTION ----------------------------------------------------
    OBJECTIVE_FUNCTION = 'Schwefel'
    plt.figure(figsize=(10, 7))
    df = pd.DataFrame()
    simulate(OBJECTIVE_FUNCTION)
    plt.savefig('figs/fss_sphere.jpg')
    df.to_csv('fss_sphere.csv')

    plt.show()


if __name__ == '__main__':
    main()
