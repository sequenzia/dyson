import layers as user_layers
from photon import models as photon_models, layers as photon_layers
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import activations, initializers, regularizers, constraints

class Model_A(photon_models.Models):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build_model(self):

        act_fn = activations.relu

        k_reg = None
        b_reg = None
        a_reg = None

        init_mu = 0.0
        init_sigma = .5

        # ----- args ----- #

        cnn_1_args = {'strides': 1,
                      'padding': 'causal',
                      'dilation_rate': 1,
                      'activation': act_fn,
                      'kernel_initializer': initializers.RandomNormal(mean=init_mu, stddev=init_sigma, seed=self.seed),
                      'use_bias': True,
                      'bias_initializer': initializers.Zeros(),
                      'kernel_regularizer': k_reg,
                      'bias_regularizer': b_reg,
                      'activity_regularizer': a_reg,
                      'trainable': True}

        dnn_out_args = {'units': 5,
                        'activation': None,
                        'use_bias': True,
                        'kernel_initializer': initializers.RandomNormal(mean=init_mu, stddev=init_sigma, seed=self.seed),
                        'bias_initializer': initializers.Zeros(),
                        'kernel_regularizer': k_reg,
                        'bias_regularizer': b_reg,
                        'activity_regularizer': a_reg,
                        'kernel_constraint': None,
                        'bias_constraint': None,
                        'trainable': True}

        self.cnn_1 = photon_layers.CNN(self.gauge,
                                       layer_nm='cnn_1',
                                       layer_args=cnn_1_args,
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

    def call(self, inputs):

        z_cnn_1 = self.cnn_1(inputs)
        z_pool = self.pool(z_cnn_1)
        z_out = self.dnn_out(z_pool)

        self.z_return = {'features': inputs,
                         'y_hat': z_out,
                         'y_true': None,
                         'x_tracking': None,
                         'y_tracking': None}

        return self.z_return

class Model_B(photon_models.Models):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build_model(self):

        flat_args = {'weights_on': True,
                     'bias_on': True}

        rd_dnn_args = {'units': 5,
                       'activation': None,
                       'use_bias': True,
                       'kernel_initializer': initializers.RandomUniform(minval=0.5, maxval=1.),
                       'bias_initializer': initializers.Zeros(),
                       'kernel_regularizer': None,
                       'bias_regularizer': None,
                       'activity_regularizer': None,
                       'kernel_constraint': constraints.NonNeg(),
                       'bias_constraint': constraints.NonNeg(),
                       'trainable': True}

        rd_config = {'branch_idx': 0,
                     'chain_idx': -1,
                     'n_inputs': 5,
                     'n_layers': 1,
                     'type': 'dnn',
                     'args': rd_dnn_args,
                     'rd_idx_type': 'model_idx',
                     'pass_on': False,
                     'trans_on': False,
                     'squeeze_on': True,
                     'softmax_on': True,
                     'logs_on': False}

        dnn_1_args = {'units': self.d_model,
                      'activation': activations.relu,
                      'use_bias': True,
                      'kernel_initializer': initializers.GlorotUniform(self.seed),
                      'bias_initializer': initializers.Zeros(),
                      'kernel_regularizer': None,
                      'bias_regularizer': None,
                      'activity_regularizer': None,
                      'kernel_constraint': None,
                      'bias_constraint': None,
                      'trainable': True}

        dnn_out_args = {'units': 5,
                        'activation': None,
                        'use_bias': False,
                        'kernel_initializer': initializers.GlorotUniform(self.seed),
                        'bias_initializer': initializers.Zeros(),
                        'kernel_regularizer': None,
                        'bias_regularizer': None,
                        'activity_regularizer': None,
                        'kernel_constraint': None,
                        'bias_constraint': None,
                        'trainable': True}

        self.build_run_data(rd_config)

        self.dnn_1 = photon_layers.DNN(self.gauge,
                                       layer_nm='dnn_1',
                                       layer_args=dnn_1_args)

        self.dnn_out = photon_layers.DNN(self.gauge,
                                         layer_nm='dnn_out',
                                         layer_args=dnn_out_args)

    def call(self, inputs):

        run_data = self.call_run_data(inputs)
        z_dnn_1 = self.dnn_1(run_data)
        z_out = self.dnn_out(z_dnn_1)

        self.z_return = {'features': inputs,
                         'y_hat': z_out,
                         'y_true': None,
                         'x_tracking': None,
                         'y_tracking': None}

        return self.z_return

class Model_C(photon_models.Models):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build_model(self):

        flat_args = {'weights_on': True,
                     'bias_on': True}

        rd_dnn_args = {'units': self.d_model,
                       'activation': None,
                       'use_bias': True,
                       'kernel_initializer': initializers.Ones(),
                       'bias_initializer': initializers.Zeros(),
                       'kernel_regularizer': None,
                       'bias_regularizer': None,
                       'activity_regularizer': None,
                       'kernel_constraint': None,
                       'bias_constraint': None,
                       'trainable': True}

        rd_config = {'branch_idx': 0,
                     'chain_idx': -1,
                     'n_inputs': self.d_model,
                     'n_layers': 0,
                     'type': 'dnn',
                     'args': rd_dnn_args,
                     'rd_idx_type': 'layer_idx',
                     'pass_on': True,
                     'trans_on': True,
                     'squeeze_on': False,
                     'softmax_on': False,
                     'logs_on': False}

        dnn_out_args = {'units': 5,
                        'activation': None,
                        'use_bias': False,
                        'kernel_initializer': initializers.GlorotUniform(self.seed),
                        'bias_initializer': initializers.Zeros(),
                        'kernel_regularizer': None,
                        'bias_regularizer': None,
                        'activity_regularizer': None,
                        'kernel_constraint': None,
                        'bias_constraint': None,
                        'trainable': True}

        self.build_run_data(rd_config)

        self.pool = photon_layers.Pool(self.gauge,
                                       layer_nm='pool',
                                       pool_type='avg',
                                       is_global=True,
                                       reg_args=None,
                                       norm_args=None)

        self.dnn_out = photon_layers.DNN(self.gauge,
                                         layer_nm='dnn_out',
                                         layer_args=dnn_out_args)

    def call(self, inputs):

        run_data = self.call_run_data(inputs)

        z_pool = self.pool(run_data)
        z_out = self.dnn_out(z_pool)

        self.z_return = {'features': inputs,
                         'y_hat': z_out,
                         'y_true': None,
                         'x_tracking': None,
                         'y_tracking': None}

        return self.z_return
