from layers import trans_layers
from photon import models as photon_models, layers as photon_layers
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import activations, initializers, regularizers, constraints
from photon.utils import args_key_chk

class Transformer_1(photon_models.Models):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build_model(self):

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

class Transformer_2(photon_models.Models):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def build_model(self, **kwargs):

        n_heads = 1
        n_blocks = 3
        split_heads = False

        drop_rate = 0

        reg_config = None
        norm_config = None

        act_fn = activations.selu

        k_reg = None # regularizers.L2()
        b_reg = None
        a_reg = None # regularizers.L1()

        # ----- configs ----- #

        bars_cnn_args = {'filters': self.d_model,
                         'kernel_size': 3,
                         'strides': 1,
                         'padding': 'causal',
                         'dilation_rate': 1,
                         'activation': act_fn,
                         'kernel_initializer': initializers.GlorotUniform(self.seed),
                         'use_bias': True,
                         'bias_initializer': initializers.Zeros(),
                         'kernel_regularizer':  k_reg,
                         'bias_regularizer': b_reg,
                         'activity_regularizer': a_reg,
                         'kernel_constraint': None,
                         'bias_constraint': None,
                         'trainable': True}

        bars_config = {'tracking_on': False,
                       'sin_pe_on': False,
                       'seq_pe_on': False,
                       'intra_pe_on': False,
                       'sub_type': 'cnn',
                       'sub_args': bars_cnn_args,
                       'n_layers': 1,
                       'output_type': 'single',
                       'kernel_size': 'units',
                       'seq_scale': 0,
                       'logs_on': False}

        out_args = {'units': 5,
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

        res_dnn_args = {'units': 5,
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

        enc_res_config = {'cols_match': False,
                          'depth_match': False,
                          'dnn_args': res_dnn_args,
                          'w_on': False,
                          'act_fn': None,
                          'init_fn': None,
                          'w_reg': None,
                          'b_reg': None,
                          'a_reg': None,
                          'pool_on': False,
                          'pool_type': None,
                          'res_drop': 0,
                          'res_norm': 0}

        attn_args = {'attn_units': self.d_model,
                     'k_init': initializers.GlorotUniform(self.seed),
                     'use_bias': True,
                     'b_init': initializers.Zeros(),
                     'act_fn': None,
                     'w_reg': None,
                     'b_reg': None,
                     'a_reg': None,
                     'logs_on': False}

        enc_block_args = {'n_heads': n_heads,
                          'd_model': self.d_model,
                          'attn_args': attn_args,
                          'split_heads': split_heads,
                          'lah_mask_on': False,
                          'log_scores': False,
                          'dff': self.d_model,
                          'res_config': enc_res_config}

        dec_block_args = {'n_heads': n_heads,
                          'd_model': self.d_model,
                          'attn_args': attn_args,
                          'split_heads': split_heads,
                          'lah_mask_on': False,
                          'log_scores': False,
                          'dff': self.d_model,
                          'res_config': enc_res_config}

        # -------- bars -------- #

        self.enc_bars = trans_layers.EncBars(self.gauge,
                                             layer_nm='enc_bars',
                                             bars_config=bars_config,
                                             reg_config=None,
                                             norm_config=None)

        self.dec_bars = trans_layers.DecBars(self.gauge,
                                             layer_nm='dec_bars',
                                             d_model=self.d_model,
                                             sin_pe_on=True,
                                             logs_on=False,
                                             reg_config=None,
                                             norm_config=None)

        # ----- encoder ----- #

        self.enc_stack = trans_layers.EncoderStack(self.gauge,
                                                   layer_nm='enc_stk',
                                                   n_blocks=n_blocks,
                                                   block_args=enc_block_args,
                                                   logs_on=False,
                                                   reg_config=reg_config,
                                                   norm_config=norm_config)

        self.dec_stack = trans_layers.DecoderStack(self.gauge,
                                                   layer_nm='dec_stack',
                                                   n_blocks=n_blocks,
                                                   block_args=dec_block_args,
                                                   logs_on=False,
                                                   reg_config=reg_config,
                                                   norm_config=norm_config)

        self.trans_out = trans_layers.TransOut(self.gauge,
                                               layer_nm='out',
                                               dnn_args=out_args,
                                               reg_config=None,
                                               norm_config=None)

    def call(self, inputs, **kwargs):

        # targets = args_key_chk(kwargs,'targets')
        # tracking = args_key_chk(kwargs, 'tracking')

        z_enc_bars = self.enc_bars(inputs)
        # z_dec_bars = self.dec_bars([targets, tracking])
        #
        # z_enc_stack = self.enc_stack(z_enc_bars)
        #
        # z_dec_stack = self.dec_stack([z_dec_bars,z_enc_stack])

        z_out = self.trans_out(z_enc_bars)

        self.z_return = {'features': inputs,
                         'y_hat': z_out,
                         'y_true': None,
                         'x_tracking': None,
                         'y_tracking': None}

        return self.z_return
