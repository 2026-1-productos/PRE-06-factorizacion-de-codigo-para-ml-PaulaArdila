"""Comparacion de modelos por su score en el conjunto de prueba."""


def compare_models(current_model, best_model, x_test, y_test):
    """Devuelve el mejor de los dos modelos segun su score."""
    if best_model is None or current_model.score(x_test, y_test) > best_model.score(
        x_test, y_test
    ):
        return current_model
    return best_model
