import layers as user_layers
from photon import models as photon_models, layers as photon_layers
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import activations, initializers, regularizers, constraints

class CNN_Base(photon_models.Models):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build_model(self):

        act_fn = activations.relu

        k_reg = None
        b_reg = None
        a_reg = None

        cnn_args = {'strides': 1,
                    'padding': 'causal',
                    'dilation_rate': 1,
                    'activation': act_fn,
                    'kernel_initializer': initializers.GlorotUniform(seed=self.seed),
                    'use_bias': True,
                    'bias_initializer': initializers.Zeros(),
                    'kernel_regularizer': k_reg,
                    'bias_regularizer': b_reg,
                    'activity_regularizer': a_reg,
                    'trainable': True}

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

        self.cnn_1 = photon_layers.CNN(self.gauge,
                                       layer_nm='cnn_1',
                                       layer_args=cnn_args,
                                       filters=self.d_model,
                                       kernel_size=5,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)
        
        self.cnn_2 = photon_layers.CNN(self.gauge,
                                       layer_nm='cnn_2',
                                       layer_args=cnn_args,
                                       filters=self.d_model,
                                       kernel_size=5,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)
        
        self.cnn_3 = photon_layers.CNN(self.gauge,
                                       layer_nm='cnn_3',
                                       layer_args=cnn_args,
                                       filters=self.d_model,
                                       kernel_size=5,
                                       reg_args=self.reg_args,
                                       norm_args=self.norm_args)

        self.cnn_4 = photon_layers.CNN(self.gauge,
                                       layer_nm='cnn_4',
                                       layer_args=cnn_args,
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

        z_cnn_1 = self.cnn_1(inputs)
        z_cnn_2 = self.cnn_2(z_cnn_1)
        z_cnn_3 = self.cnn_3(z_cnn_2)
        z_cnn_4 = self.cnn_4(z_cnn_3)
        z_pool = self.pool(z_cnn_4)
        z_out = self.dnn_out(z_pool)

        self.z_return = {'features': inputs,
                         'y_hat': z_out,
                         'y_true': None,
                         'x_tracking': None,
                         'y_tracking': None}

        return self.z_return
