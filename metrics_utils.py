import numpy as np

class Metrics:
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def accuracy_within_tolerance(y_true, y_pred, tolerance=1):
        """
        Pontosság: Hány predikció esik a valós érték körüli toleranciába.
        :param y_true: Valós értékek
        :param y_pred: Predikciók
        :param tolerance: Toleranciaérték
        """
        return np.mean(np.abs(y_true - y_pred) <= tolerance) * 100
