from dyson import layers as user_layers
from photon import models as photon_models, layers as photon_layers
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import activations, initializers, regularizers, constraints


class DNN_Base(photon_models.Models):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build_model(self):

        act_fn = activations.relu

        k_reg = regularizers.L1(l1=0.01)
        b_reg = None
        a_reg = None

        dnn_args = {'units': 5,
                    'activation': activations.softmax,
                    'use_bias': True,
                    'kernel_initializer': initializers.GlorotUniform(seed=self.seed),
                    'bias_initializer': initializers.Zeros(),
                    'kernel_regularizer': k_reg,
                    'bias_regularizer': b_reg,
                    'activity_regularizer': a_reg,
                    'kernel_constraint': None,
                    'bias_constraint': None,
                    'trainable': True}

        dnn_out_args = {'units': 5,
                        'activation': activations.softmax,
                        'use_bias': True,
                        'kernel_initializer': initializers.GlorotUniform(seed=self.seed),
                        'bias_initializer': initializers.Zeros(),
                        'kernel_regularizer': k_reg,
                        'bias_regularizer': b_reg,
                        'activity_regularizer': a_reg,
                        'kernel_constraint': None,
                        'bias_constraint': None,
                        'trainable': True}

        self.dnn_1 = photon_layers.DNN(self.gauge,
                                       layer_nm='dnn_1',
                                       layer_args=dnn_args,
                                       filters=self.d_model,
                                       kernel_size=5,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)

        self.dnn_2 = photon_layers.DNN(self.gauge,
                                       layer_nm='dnn_2',
                                       layer_args=dnn_args,
                                       filters=self.d_model,
                                       kernel_size=5,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)

        self.dnn_3 = photon_layers.DNN(self.gauge,
                                       layer_nm='dnn_3',
                                       layer_args=dnn_args,
                                       filters=self.d_model,
                                       kernel_size=5,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)

        self.pool = photon_layers.Pool(self.gauge,
                                       layer_nm='pool',
                                       pool_type='avg',
                                       is_global=True,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)

        self.dnn_out = photon_layers.DNN(self.gauge,
                                         layer_nm='dnn_out',
                                         layer_args=dnn_out_args,
                                         reg_args=None,
                                         norm_args=None)

    def call(self, inputs, **kwargs):

        z_dnn_1 = self.dnn_1(inputs)
        z_dnn_2 = self.dnn_2(z_dnn_1)
        z_dnn_3 = self.dnn_3(z_dnn_2)
        z_pool = self.pool(z_dnn_3)
        z_out = self.dnn_out(z_pool)

        self.z_return = {'features': inputs,
                         'y_hat': z_out,
                         'y_true': None,
                         'x_tracking': None,
                         'y_tracking': None}

        return self.z_return
