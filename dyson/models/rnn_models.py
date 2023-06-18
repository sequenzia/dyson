from dyson import layers as user_layers
from photon import models as photon_models, layers as photon_layers
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import activations, initializers, regularizers, constraints

class LSTM_Pool(photon_models.Models):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build_model(self, **kwargs):

        act_fn = activations.relu

        k_reg = None
        b_reg = None
        a_reg = None

        rnn_args = {'units': self.d_model,
                      'activation': activations.tanh,
                      'recurrent_activation': 'sigmoid',
                      'kernel_initializer': initializers.GlorotUniform(seed=self.seed),
                      'recurrent_initializer': initializers.Orthogonal(seed=self.seed),
                      'use_bias': True,
                      'bias_initializer': initializers.Zeros(),
                      'kernel_regularizer': k_reg,
                      'recurrent_regularizer': k_reg,
                      'bias_regularizer': b_reg,
                      'dropout': self.drop_rate,
                      'stateful': False,
                      'unroll': False,
                      'return_sequences': True,
                      'return_state': False}

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

        self.rnn_1 = photon_layers.RNN(self.gauge,
                                       layer_nm='rnn_1',
                                       rnn_args=rnn_args,
                                       rnn_type='lstm',
                                       mask_on=False,
                                       reset_type=None,
                                       logs_on=False,
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

        z_rnn_1 = self.rnn_1(inputs)
        z_pool = self.pool(z_rnn_1)
        z_out = self.dnn_out(z_pool)

        self.z_return = {'features': inputs,
                         'y_hat': z_out,
                         'y_true': None,
                         'x_tracking': None,
                         'y_tracking': None}

        return self.z_return

class LSTM_NoPool(photon_models.Models):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build_model(self, **kwargs):

        act_fn = activations.relu

        k_reg = None
        b_reg = None
        a_reg = None

        rnn_args = {'units': self.d_model,
                    'activation': activations.tanh,
                    'recurrent_activation': 'sigmoid',
                    'kernel_initializer': initializers.GlorotUniform(seed=self.seed),
                    'recurrent_initializer': initializers.Orthogonal(seed=self.seed),
                    'use_bias': True,
                    'bias_initializer': initializers.Zeros(),
                    'kernel_regularizer': k_reg,
                    'recurrent_regularizer': k_reg,
                    'bias_regularizer': b_reg,
                    'dropout': self.drop_rate,
                    'stateful': False,
                    'unroll': False,
                    'return_sequences': True,
                    'return_state': False}

        rnn_out_args = {'units': self.d_model,
                      'activation': activations.tanh,
                      'recurrent_activation': 'sigmoid',
                      'kernel_initializer': initializers.GlorotUniform(seed=self.seed),
                      'recurrent_initializer': initializers.Orthogonal(seed=self.seed),
                      'use_bias': True,
                      'bias_initializer': initializers.Zeros(),
                      'kernel_regularizer': k_reg,
                      'recurrent_regularizer': k_reg,
                      'bias_regularizer': b_reg,
                      'dropout': self.drop_rate,
                      'stateful': False,
                      'unroll': False,
                      'return_sequences': False,
                      'return_state': False}

        dnn_out_args = {'units': 5,
                        'activation': None,
                        'use_bias': True,
                        'kernel_initializer': initializers.GlorotUniform(seed=self.seed),
                        'bias_initializer': initializers.Zeros(),
                        'kernel_regularizer': k_reg,
                        'bias_regularizer': b_reg,
                        'activity_regularizer': a_reg,
                        'kernel_constraint': None,
                        'bias_constraint': None,
                        'trainable': True}

        self.rnn_1 = photon_layers.RNN(self.gauge,
                                       layer_nm='rnn_1',
                                       rnn_args=rnn_args,
                                       rnn_type='lstm',
                                       mask_on=False,
                                       reset_type=None,
                                       logs_on=False,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)

        self.rnn_out = photon_layers.RNN(self.gauge,
                                       layer_nm='rnn_out',
                                       rnn_args=rnn_out_args,
                                       rnn_type='lstm',
                                       mask_on=False,
                                       reset_type=None,
                                       logs_on=False,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)

        self.dnn_out = photon_layers.DNN(self.gauge,
                                         layer_nm='dnn_out',
                                         layer_args=dnn_out_args,
                                         reg_args=None,
                                         norm_args=None)

    def call(self, inputs, **kwargs):

        z_rnn_1 = self.rnn_1(inputs)
        z_rnn_out = self.rnn_out(z_rnn_1)
        z_out = self.dnn_out(z_rnn_out)

        self.z_return = {'features': inputs,
                         'y_hat': z_out,
                         'y_true': None,
                         'x_tracking': None,
                         'y_tracking': None}

        return self.z_return

class LSTM_Deep(photon_models.Models):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build_model(self, **kwargs):

        act_fn = activations.relu

        k_reg = None
        b_reg = None
        a_reg = None

        rnn_args = {'units': self.d_model,
                    'activation': activations.tanh,
                    'recurrent_activation': 'sigmoid',
                    'kernel_initializer': initializers.GlorotUniform(seed=self.seed),
                    'recurrent_initializer': initializers.Orthogonal(seed=self.seed),
                    'use_bias': True,
                    'bias_initializer': initializers.Zeros(),
                    'kernel_regularizer': k_reg,
                    'recurrent_regularizer': k_reg,
                    'bias_regularizer': b_reg,
                    'dropout': self.drop_rate,
                    'stateful': False,
                    'unroll': False,
                    'return_sequences': True,
                    'return_state': False}

        rnn_out_args = {'units': self.d_model,
                      'activation': activations.tanh,
                      'recurrent_activation': 'sigmoid',
                      'kernel_initializer': initializers.GlorotUniform(seed=self.seed),
                      'recurrent_initializer': initializers.Orthogonal(seed=self.seed),
                      'use_bias': True,
                      'bias_initializer': initializers.Zeros(),
                      'kernel_regularizer': k_reg,
                      'recurrent_regularizer': k_reg,
                      'bias_regularizer': b_reg,
                      'dropout': self.drop_rate,
                      'stateful': False,
                      'unroll': False,
                      'return_sequences': False,
                      'return_state': False}

        dnn_out_args = {'units': 5,
                        'activation': None,
                        'use_bias': True,
                        'kernel_initializer': initializers.GlorotUniform(seed=self.seed),
                        'bias_initializer': initializers.Zeros(),
                        'kernel_regularizer': k_reg,
                        'bias_regularizer': b_reg,
                        'activity_regularizer': a_reg,
                        'kernel_constraint': None,
                        'bias_constraint': None,
                        'trainable': True}

        self.rnn_1 = photon_layers.RNN(self.gauge,
                                       layer_nm='rnn_1',
                                       rnn_args=rnn_args,
                                       rnn_type='lstm',
                                       mask_on=False,
                                       reset_type=None,
                                       logs_on=False,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)

        self.rnn_2 = photon_layers.RNN(self.gauge,
                                       layer_nm='rnn_2',
                                       rnn_args=rnn_args,
                                       rnn_type='lstm',
                                       mask_on=False,
                                       reset_type=None,
                                       logs_on=False,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)


        self.rnn_3 = photon_layers.RNN(self.gauge,
                                       layer_nm='rnn_3',
                                       rnn_args=rnn_args,
                                       rnn_type='lstm',
                                       mask_on=False,
                                       reset_type=None,
                                       logs_on=False,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)


        self.rnn_4 = photon_layers.RNN(self.gauge,
                                       layer_nm='rnn_4',
                                       rnn_args=rnn_args,
                                       rnn_type='lstm',
                                       mask_on=False,
                                       reset_type=None,
                                       logs_on=False,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)

        self.rnn_5 = photon_layers.RNN(self.gauge,
                                       layer_nm='rnn_5',
                                       rnn_args=rnn_args,
                                       rnn_type='lstm',
                                       mask_on=False,
                                       reset_type=None,
                                       logs_on=False,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)

        self.rnn_out = photon_layers.RNN(self.gauge,
                                       layer_nm='rnn_out',
                                       rnn_args=rnn_out_args,
                                       rnn_type='lstm',
                                       mask_on=False,
                                       reset_type=None,
                                       logs_on=False,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)

        self.dnn_out = photon_layers.DNN(self.gauge,
                                         layer_nm='dnn_out',
                                         layer_args=dnn_out_args,
                                         reg_args=None,
                                         norm_args=None)

    def call(self, inputs, **kwargs):

        z_rnn_1 = self.rnn_1(inputs)
        z_rnn_2 = self.rnn_2(z_rnn_1)
        z_rnn_3 = self.rnn_3(z_rnn_2)
        z_rnn_4 = self.rnn_4(z_rnn_3)
        z_rnn_5 = self.rnn_5(z_rnn_4)
        z_rnn_out = self.rnn_out(z_rnn_5)
        z_out = self.dnn_out(z_rnn_out)

        self.z_return = {'features': inputs,
                         'y_hat': z_out,
                         'y_true': None,
                         'x_tracking': None,
                         'y_tracking': None}

        return self.z_return


