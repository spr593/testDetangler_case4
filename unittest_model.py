import unittest
from model import *
from sklearn.base import accuracy_score
from sklearn.metrics import f1_score


class TestStackingModel(unittest.TestCase):
    def __init__(self, model, y_val, y_test):
        self.model = model
        self.y_val = y_val
        self.y_test = y_test

    def test_model_performance(self, validation_predictions, predictions):
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(validation_predictions)

        # Check if the predictions have the correct shape
        self.assertEqual(predictions.shape, self.y_val.shape)
        self.assertEqual(validation_predictions.shape, self.y_val.shape)

        # Check for f1
        test_f1 = f1_score(self.y_test, predictions)
        validation_f1 = f1_score(self.y_val, validation_predictions)
        self.assertGreater(test_f1, 0.8)  # Example threshold
        self.assertGreater(validation_f1, 0.9)  # Example threshold

