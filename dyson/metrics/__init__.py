import tensorflow as tf
import numpy as np


def convert_logits(data, config):
    # if 'from_logits' in config:
    #     if not config['from_logits']:
    #         return tf.nn.softmax(data)
    #     else:
    #         return data
    # else:
    #     return tf.nn.softmax(data)
    return data

class Metrics():

    def __init__(self):
        self.AUC = AUC
        self.TruePositives = TruePositives
        self.TrueNegatives = TrueNegatives
        self.FalsePositives = FalsePositives
        self.FalseNegatives = FalseNegatives
        self.Precision = Precision
        self.Recall = Recall
        self.PrecisionAtRecall = PrecisionAtRecall
        self.SensitivityAtSpecificity = SensitivityAtSpecificity
        self.SpecificityAtSensitivity = SpecificityAtSensitivity

class AUC():

    def __init__(self, config):

        self.config = config

        self.main_acc = tf.keras.metrics.AUC(**self.config)
        self.val_acc = tf.keras.metrics.AUC(**self.config)

    def __call__(self, y_true, y_hat, run_type):

        if run_type != 'val':

            self.main_acc.update_state(y_true, y_hat)

            return self.main_acc.result().numpy()

        if run_type == 'val':

            self.val_acc.update_state(y_true, y_hat)

            return self.val_acc.result().numpy()

class TruePositives():

    def __init__(self, config):

        self.config = config

        self.main_acc = tf.keras.metrics.TruePositives(**self.config)
        self.val_acc = tf.keras.metrics.TruePositives(**self.config)

    def __call__(self, y_true, y_hat, run_type):

        y_hat = convert_logits(y_hat, self.config)

        if run_type != 'val':

            self.main_acc.update_state(y_true, y_hat)

            return self.main_acc.result().numpy()

        if run_type == 'val':

            self.val_acc.update_state(y_true, y_hat)

            return self.val_acc.result().numpy()

class TrueNegatives():

    def __init__(self, config):

        self.config = config

        self.main_acc = tf.keras.metrics.TrueNegatives(**self.config)
        self.val_acc = tf.keras.metrics.TrueNegatives(**self.config)

    def __call__(self, y_true, y_hat, run_type):

        y_hat = convert_logits(y_hat, self.config)

        if run_type != 'val':

            self.main_acc.update_state(y_true, y_hat)

            return self.main_acc.result().numpy()

        if run_type == 'val':

            self.val_acc.update_state(y_true, y_hat)

            return self.val_acc.result().numpy()

class FalsePositives():

    def __init__(self, config):

        self.config = config

        self.main_acc = tf.keras.metrics.FalsePositives(**self.config)
        self.val_acc = tf.keras.metrics.FalsePositives(**self.config)

    def __call__(self, y_true, y_hat, run_type):

        y_hat = convert_logits(y_hat, self.config)

        if run_type != 'val':

            self.main_acc.update_state(y_true, y_hat)

            return self.main_acc.result().numpy()

        if run_type == 'val':

            self.val_acc.update_state(y_true, y_hat)

            return self.val_acc.result().numpy()

class FalseNegatives():

    def __init__(self, config):

        self.config = config

        self.main_acc = tf.keras.metrics.FalseNegatives(**self.config)
        self.val_acc = tf.keras.metrics.FalseNegatives(**self.config)

    def __call__(self, y_true, y_hat, run_type):

        y_hat = convert_logits(y_hat, self.config)

        if run_type != 'val':

            self.main_acc.update_state(y_true, y_hat)

            return self.main_acc.result().numpy()

        if run_type == 'val':

            self.val_acc.update_state(y_true, y_hat)

            return self.val_acc.result().numpy()

class Precision():

    def __init__(self, config):

        self.config = config

        self.main_acc = tf.keras.metrics.Precision(**self.config)
        self.val_acc = tf.keras.metrics.Precision(**self.config)

    def __call__(self, y_true, y_hat, run_type):

        y_hat = convert_logits(y_hat, self.config)

        if run_type != 'val':

            self.main_acc.update_state(y_true, y_hat)

            return self.main_acc.result().numpy()

        if run_type == 'val':

            self.val_acc.update_state(y_true, y_hat)

            return self.val_acc.result().numpy()

class Recall():

    def __init__(self, config):

        self.config = config

        self.main_acc = tf.keras.metrics.Recall(**self.config)
        self.val_acc = tf.keras.metrics.Recall(**self.config)

    def __call__(self, y_true, y_hat, run_type):

        y_hat = convert_logits(y_hat, self.config)

        if run_type != 'val':

            self.main_acc.update_state(y_true, y_hat)

            return self.main_acc.result().numpy()

        if run_type == 'val':

            self.val_acc.update_state(y_true, y_hat)

            return self.val_acc.result().numpy()

class PrecisionAtRecall():

    def __init__(self, config):

        self.config = config

        self.main_acc = tf.keras.metrics.PrecisionAtRecall(**self.config)
        self.val_acc = tf.keras.metrics.PrecisionAtRecall(**self.config)

    def __call__(self, y_true, y_hat, run_type):

        y_hat = convert_logits(y_hat, self.config)

        if run_type != 'val':

            self.main_acc.update_state(y_true, y_hat)

            return self.main_acc.result().numpy()

        if run_type == 'val':

            self.val_acc.update_state(y_true, y_hat)

            return self.val_acc.result().numpy()

class SensitivityAtSpecificity():

    def __init__(self, config):

        self.config = config

        self.main_acc = tf.keras.metrics.SensitivityAtSpecificity(**self.config)
        self.val_acc = tf.keras.metrics.SensitivityAtSpecificity(**self.config)

    def __call__(self, y_true, y_hat, run_type):

        y_hat = convert_logits(y_hat, self.config)

        if run_type != 'val':

            self.main_acc.update_state(y_true, y_hat)

            return self.main_acc.result().numpy()

        if run_type == 'val':

            self.val_acc.update_state(y_true, y_hat)

            return self.val_acc.result().numpy()

class SpecificityAtSensitivity():

    def __init__(self, config):

        self.config = config

        self.main_acc = tf.keras.metrics.SpecificityAtSensitivity(**self.config)
        self.val_acc = tf.keras.metrics.SpecificityAtSensitivity(**self.config)

    def __call__(self, y_true, y_hat, run_type):

        y_hat = convert_logits(y_hat, self.config)

        if run_type != 'val':

            self.main_acc.update_state(y_true, y_hat)

            return self.main_acc.result().numpy()

        if run_type == 'val':

            self.val_acc.update_state(y_true, y_hat)

            return self.val_acc.result().numpy()

