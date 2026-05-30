#
# Busque los mejores parametros de un modelo ElasticNet para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere los siguientes valores de los hiperparametros y obtenga el
# mejor modelo (alpha, l1_ratio):
#    (0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.1, 0.05), (0.3, 0.2)
#

from sklearn.linear_model import ElasticNet

from ._internals.run_experiment import run_experiment

PARAMS = [(0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.1, 0.05), (0.3, 0.2)]


def main():
    for alpha, l1_ratio in PARAMS:
        estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12345)
        run_experiment(estimator)


if __name__ == "__main__":
    main()
