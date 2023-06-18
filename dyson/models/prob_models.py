from dyson import layers as user_layers
from photon import models as photon_models, layers as photon_layers, kernels as photon_kernels

from tensorflow.keras import layers as tf_layers
from tensorflow.keras import activations, initializers, regularizers, constraints

import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib

tfd = tfp.distributions
tfpl = tfp.layers
tfpl_utils = tfpl.util
tfb = tfp.bijectors

class Prob_1(photon_models.Models):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build_model(self):

        act_fn = activations.relu

        k_reg = None
        b_reg = None
        a_reg = None

        k_prior_fn = tfpl.default_multivariate_normal_fn

        k_post_fn = tfp.layers.default_mean_field_normal_fn(is_singular=False,
                                                            loc_initializer=initializers.random_normal(stddev=0.1),
                                                            untransformed_scale_initializer=initializers.random_normal(mean=-3.0, stddev=0.1),
                                                            loc_regularizer=None,
                                                            untransformed_scale_regularizer=None,
                                                            loc_constraint=None,
                                                            untransformed_scale_constraint=None)

        b_post_fn = tfp.layers.default_mean_field_normal_fn(is_singular=True,
                                                            loc_initializer=initializers.random_normal(stddev=0.1),
                                                            untransformed_scale_initializer=initializers.random_normal(mean=-3.0, stddev=0.1),
                                                            loc_regularizer=None,
                                                            untransformed_scale_regularizer=None,
                                                            loc_constraint=None,
                                                            untransformed_scale_constraint=None)

        dnn_args = {'units':32,
                    'activation':act_fn,
                    'activity_regularizer':None,
                    'trainable':True,
                    'kernel_posterior_fn':k_post_fn,
                    'kernel_posterior_tensor_fn':(lambda d: d.sample()),
                    'kernel_prior_fn':k_prior_fn,
                    'kernel_divergence_fn':(lambda q, p, ignore: kl_lib.kl_divergence(q, p)),
                    'bias_posterior_fn':b_post_fn,
                    'bias_posterior_tensor_fn':(lambda d: d.sample()),
                    'bias_prior_fn':None,
                    'bias_divergence_fn':(lambda q, p, ignore: kl_lib.kl_divergence(q, p))}

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

        self.dnn_1 = photon_layers.Base(self.gauge, layer_nm='dnn_1', layer=tfpl.DenseFlipout(**dnn_args, name='dnn_1'))
        self.dnn_2 = photon_layers.Base(self.gauge, layer_nm='dnn_2', layer=tfpl.DenseFlipout(**dnn_args, name='dnn_2'))

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
        z_pool = self.pool(z_dnn_2)
        z_out = self.dnn_out(z_pool)

        self.z_return = {'features': inputs,
                         'y_hat': z_out,
                         'y_true': None,
                         'x_tracking': None,
                         'y_tracking': None}

        return self.z_return

class Prob_2(photon_models.Models):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build_model(self):

        # kernels = photon_kernels.Kernels(self.gauge)

        act_fn = activations.relu

        k_reg = None
        b_reg = None
        a_reg = None

        k_prior_fn = tfpl.default_multivariate_normal_fn

        k_post_fn = tfp.layers.default_mean_field_normal_fn(is_singular=False,
                                                            loc_initializer=initializers.random_normal(stddev=0.1),
                                                            untransformed_scale_initializer=initializers.random_normal(mean=-3.0, stddev=0.1),
                                                            loc_regularizer=None,
                                                            untransformed_scale_regularizer=None,
                                                            loc_constraint=None,
                                                            untransformed_scale_constraint=None)

        b_post_fn = tfp.layers.default_mean_field_normal_fn(is_singular=True,
                                                            loc_initializer=initializers.random_normal(stddev=0.1),
                                                            untransformed_scale_initializer=initializers.random_normal(mean=-3.0, stddev=0.1),
                                                            loc_regularizer=None,
                                                            untransformed_scale_regularizer=None,
                                                            loc_constraint=None,
                                                            untransformed_scale_constraint=None)

        dnn_args = {'units':32,
                    'activation':None,
                    'activity_regularizer':None,
                    'trainable':True,
                    'kernel_posterior_fn':k_post_fn,
                    'kernel_posterior_tensor_fn':(lambda d: d.sample()),
                    'kernel_prior_fn':k_prior_fn,
                    'kernel_divergence_fn':(lambda q, p, ignore: kl_lib.kl_divergence(q, p)),
                    'bias_posterior_fn':b_post_fn,
                    'bias_posterior_tensor_fn':(lambda d: d.sample()),
                    'bias_prior_fn':None,
                    'bias_divergence_fn':(lambda q, p, ignore: kl_lib.kl_divergence(q, p))}


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

        # dnn = tfpl.DenseVariational(units=self.d_model,
        #                             make_posterior_fn=kernels.posterior_mean_field,
        #                             make_prior_fn=kernels.prior_trainable)




        dnn = tfpl.DenseFlipout(**dnn_args)


        # main_out = tfpl.IndependentBernoulli(2,
        #                                      convert_to_tensor_fn=tfd.Distribution.sample,
        #                                      validate_args=True)

        self.dnn_1 = photon_layers.Base(self.gauge, layer_nm='dnn_1', layer=dnn)

        # self.dnn_1 = photon_layers.DNN(self.gauge,
        #                                layer_nm='dnn_1',
        #                                layer_args=dnn_args,
        #                                reg_args=None,
        #                                norm_args=None)

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
        z_pool = self.pool(z_dnn_1)
        z_out = self.dnn_out(z_pool)

        self.z_return = {'features': inputs,
                         'y_hat': z_out,
                         'y_true': None,
                         'x_tracking': None,
                         'y_tracking': None}

        return self.z_return
