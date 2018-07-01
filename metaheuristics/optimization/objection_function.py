from __future__ import division

import numpy as np

from scipy import optimize
from deap.benchmarks import schwefel
from deap.benchmarks import griewank
from deap.benchmarks import schaffer

from abc import ABCMeta
from abc import abstractmethod
from six import add_metaclass

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from keras.models import Sequential
from keras import layers

@add_metaclass(ABCMeta)
class ObjectiveFunction(object):

    def __init__(self, name, dim, minf, maxf):
        self.name = name
        self.dim = dim
        self.minf = minf
        self.maxf = maxf

    def sample(self):
        return np.random.uniform(low=self.minf, high=self.maxf, size=self.dim)

    def custom_sample(self):
        return np.repeat(self.minf, repeats=self.dim) \
               + np.random.uniform(low=0, high=1, size=self.dim) *\
               np.repeat(self.maxf - self.minf, repeats=self.dim)

    @abstractmethod
    def evaluate(self, x):
        pass


@add_metaclass(ABCMeta)
class PartitionalClusteringObjectiveFunction(ObjectiveFunction):

    def __init__(self, dim, n_clusters, data):
        super(PartitionalClusteringObjectiveFunction, self)\
            .__init__('PartitionalClusteringObjectiveFunction', dim, 0.0, 1.0)
        self.n_clusters = n_clusters
        self.centroids = {}
        self.data = data

    def decode(self, x):
        centroids = x.reshape(self.n_clusters, self.data.shape[1])
        self.centroids = dict(enumerate(centroids))

    @abstractmethod
    def evaluate(self, x):
        pass


class QuantizationError(PartitionalClusteringObjectiveFunction):

    def __init__(self, dim, n_clusters, data):
        super(QuantizationError, self).__init__(dim, n_clusters, data)
        self.name = 'QuantizationError'

    def evaluate(self, x):
        self.decode(x)

        clusters = {key: [] for key in self.centroids.keys()}
        for instance in self.data:
            distances = [np.linalg.norm(self.centroids[idx] - instance)
                         for idx in self.centroids]
            clusters[np.argmin(distances)].append(instance)

        outer_sum = 0.0
        for centroid in self.centroids:
            inner_sum = 0.0
            if len(clusters[centroid]) > 0:
                for instance in clusters[centroid]:
                    inner_sum += np.linalg.norm(instance - self.centroids[centroid])
                inner_sum /= len(clusters[centroid])
            outer_sum += inner_sum
        return outer_sum/len(self.centroids)


class SumOfSquaredErrors(PartitionalClusteringObjectiveFunction):

    def __init__(self, dim, n_clusters, data):
        super(SumOfSquaredErrors, self).__init__(dim, n_clusters, data)
        self.name = 'SumOfSquaredErrors'

    def evaluate(self, x):
        self.decode(x)

        clusters = {key: [] for key in self.centroids.keys()}
        for instance in self.data:
            distances = [np.linalg.norm(self.centroids[idx] - instance)
                         for idx in self.centroids]
            clusters[np.argmin(distances)].append(instance)

        sum_of_squared_errors = 0.0
        for idx in self.centroids:
            distances = [np.linalg.norm(self.centroids[idx] - instance)
                         for instance in clusters[idx]]
            sum_of_squared_errors += sum(np.power(distances, 2))
        return sum_of_squared_errors


class DeepNeuralNetworkObjectiveFunction(ObjectiveFunction):

    def __init__(self, dim):
        super(DeepNeuralNetworkObjectiveFunction, self).__init__('DeepNeuralNetworkObjectiveFunction', dim, 0, 2)
        digits = load_digits()
        self.data = digits['data']
        self.target = digits['target']

    def evaluate(self, x):
        model = Sequential()
        model.add(layers.Dense(12, input_dim=self.data.shape[1], activation='relu'))
        print('evaluating ...')
        for idx, layer in enumerate(np.round(x)):
            if layer == 0:
                model.add(layers.LSTM(150, return_sequences=True))
                print('adding layer LSTM ...')
            elif layer == 1:
                model.add(layers.MaxPooling2D())
                print('adding layer MaxPooling ...')
            elif layer == 2:
                model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
                print('adding layer Conv2D ...')
        model.add(layers.Dense(1, activation='softmax'))

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        train_x, test_x, train_y, test_y = train_test_split(self.data, self.target, stratify=self.target)
        results = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))
        return -1 * results.history["val_acc"]


class Sphere(ObjectiveFunction):

    def __init__(self, dim):
        super(Sphere, self).__init__('Sphere', dim, -100.0, 100.0)

    def evaluate(self, x):
        return sum(np.power(x, 2))


class Rosenbrock(ObjectiveFunction):

    def __init__(self, dim):
        super(Rosenbrock, self).__init__('Rosenbrock', dim, -30.0, 30.0)

    def evaluate(self, x):
        return optimize.rosen(x)


class Rastrigin(ObjectiveFunction):

    def __init__(self, dim):
        super(Rastrigin, self).__init__('Rastrigin', dim, -5.12, 5.12)

    def evaluate(self, x):
        return 10 * len(x)\
               + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * np.array(x)))


class Schwefel(ObjectiveFunction):

    def __init__(self, dim):
        super(Schwefel, self).__init__('Schwefel', dim, -500.0, 500.0)

    def evaluate(self, x):
        return schwefel(x)[0]


class Griewank(ObjectiveFunction):

    def __init__(self, dim):
        super(Griewank, self).__init__('Griewank', dim, -600.0, 600.0)

    def evaluate(self, x):
        return griewank(x)[0]


class Schaffer(ObjectiveFunction):

    def __init__(self, dim):
        super(Schaffer, self).__init__('Schaffer', dim, -100.0, 100.0)

    def evaluate(self, x):
        return schaffer(x)[0]

