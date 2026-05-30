#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#

from sklearn.neighbors import KNeighborsRegressor

from ._internals.run_experiment import run_experiment

NEIGHBORS = [3, 5, 7, 9, 11]


def main():
    for n in NEIGHBORS:
        estimator = KNeighborsRegressor(n_neighbors=n)
        run_experiment(estimator)


if __name__ == "__main__":
    main()
