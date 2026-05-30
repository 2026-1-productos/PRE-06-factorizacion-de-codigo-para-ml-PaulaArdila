"""Carga y preparacion de los datos de calidad del vino tinto."""

from pathlib import Path

import pandas as pd

DATA_FOLDER = Path("data/winequality-red")


def prepare_data():
    """Carga los conjuntos de entrenamiento y prueba desde los CSV locales."""
    x_train = pd.read_csv(DATA_FOLDER / "x_train.csv")
    x_test = pd.read_csv(DATA_FOLDER / "x_test.csv")
    y_train = pd.read_csv(DATA_FOLDER / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_FOLDER / "y_test.csv").squeeze("columns")
    return x_train, x_test, y_train, y_test
