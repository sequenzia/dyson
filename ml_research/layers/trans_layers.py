from photon import layers as photon_layers
import tensorflow as tf
from tensorflow.keras import layers as tf_layers
from photon.utils import args_key_chk

class EncBars(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, bars_config, **kwargs):
        """
        Layer that produces bars from inputs.

        bars_config:

            n_layers (int 0,1,2):
                0: Create n_layers based on the depth of the seq dim. One layer for each record in the seq_dim
                1: One single layer
                2: Two layers; one for the past records (seq_dim[:-1]) and one layer for the cur records (seq_dim[-1:])

            output_type (string: all, single, split_a, split_b) :
                all: One-to-one output for each internal layer
                single: Single output with all internal layers merged
                split_a: Two outputs; One with all past layers (seq_dim[:-1]) & One with cur layer (seq_dim[-1:])
                split_b: Two outputs; One with all layers & One with cur layer (seq_dim[-1:])

            kernel_size (string: units, cols):
                units: the kernel size and output cols of the internal layers will be set to the units in sub_args
                cols:  the kernel size and output cols of the internal layers will match the num of cols in the last dim of the input

            seq_scale (float) (should set to around .1):
                if !=0: Adds a learnable scaling var for each internal layer. Inits a constant as float(((idx + 1) * seq_scale) + 1)

                constraints: n_layers != 1

        """
        super().__init__(gauge, layer_nm, **kwargs)

        self.tracking_on = bars_config['tracking_on']
        self.sin_pe_on = bars_config['sin_pe_on']
        self.seq_pe_on = bars_config['seq_pe_on']
        self.intra_pe_on = bars_config['intra_pe_on']

        self.sub_args = bars_config['sub_args']
        self.sub_type = bars_config['sub_type']
        self.n_layers = bars_config['n_layers']
        self.output_type = bars_config['output_type']
        self.kernel_size = bars_config['kernel_size']
        self.seq_scale = bars_config['seq_scale']

        self.sub_norm_on = args_key_chk(bars_config, 'sub_norm_on', False)
        self.sub_reg_on = args_key_chk(bars_config, 'sub_norm_on', False)

        self.logs_on = bars_config['logs_on']

        self.seq_scaling_on = False

        if self.seq_scale != 0:
            self.seq_scaling_on = True

        self.subs_reg_args = self.reg_args
        self.subs_norm_args = self.norm_args

        if not self.sub_reg_on:
            self.subs_reg_args = None

        if not self.sub_norm_on:
            self.subs_norm_args = None

        self.seq_split = False
        self.sub_layers = []

    def build(self, input_shp):

        self.input_shp = input_shp

        if self.tracking_on:
            self.input_shp = input_shp[0]

        self.batch_size = self.input_shp[0]
        self.seq_depth = self.input_shp[1]
        self.n_cols = self.input_shp[-1]

        self.gauge_dtype = self.gauge.tree.data.dtype

        if self.n_layers == 0:
            self.n_layers = self.input_shp[1]
            self.seq_split = True

        if self.kernel_size == 'cols':
            self.sub_args['units'] = self.input_shp[-1]

        self.scaling_vars = []

        if self.seq_split:

            for idx in range(self.n_layers):

                nm = self.layer_nm + '_dnn_' + str(idx)

                if self.sub_type == 'dnn':
                    _layer = photon_layers.DNN(self.gauge,
                                               layer_nm=self.layer_nm + '_dnn_' + str(idx),
                                               layer_args=self.sub_args,
                                               reg_args=self.subs_reg_args,
                                               norm_args=self.subs_norm_args,
                                               is_child=True)

                if self.sub_type == 'cnn':
                    _layer = CNN(
                        self.gauge,
                        layer_nm=self.layer_nm + '_cnn_' + str(idx),
                        layer_args=self.sub_args,
                        reg_args=self.subs_reg_args,
                        norm_args=self.subs_norm_args,
                        is_child=True)

                self.sub_layers.append(_layer)

                if self.seq_scaling_on:

                    scale_val = float(((idx + 1) * self.seq_scale) + 1)

                    scale_init = tf.keras.initializers.Constant(scale_val)

                    scale_var = self.add_weight(initializer=scale_init, name='seq_scale_' + nm, dtype=self.gauge_dtype)

                    self.scaling_vars.append(scale_var)

        if not self.seq_split:

            if self.n_layers == 1:

                if self.sub_type == 'dnn':
                    _layer = photon_layers.DNN(self.gauge,
                                               layer_nm=self.layer_nm + '_dnn_0',
                                               layer_args=self.sub_args,
                                               reg_args=self.subs_reg_args,
                                               norm_args=self.subs_norm_args,
                                               is_child=True)

                if self.sub_type == 'cnn':
                    _layer = photon_layers.CNN(self.gauge,
                                               layer_nm=self.layer_nm + '_cnn_0',
                                               layer_args=self.sub_args,
                                               reg_args=self.subs_reg_args,
                                               norm_args=self.subs_norm_args,
                                               is_child=True)

                self.sub_layers.append(_layer)

            if self.n_layers == 2:

                if self.sub_type == 'dnn':
                    _layer_0 = photon_layers.DNN(self.gauge,
                                                 layer_nm=self.layer_nm + '_past',
                                                 layer_args=self.sub_args,
                                                 reg_args=self.subs_reg_args,
                                                 norm_args=self.subs_norm_args,
                                                 is_child=True)

                    _layer_1 = photon_layers.DNN(self.gauge,
                                                 layer_nm=self.layer_nm + '_cur',
                                                 layer_args=self.sub_args,
                                                 reg_args=self.subs_reg_args,
                                                 norm_args=self.subs_norm_args,
                                                 is_child=True)

                if self.sub_type == 'cnn':
                    _layer_0 = CNN(
                        self.gauge,
                        layer_nm=self.layer_nm + '_past',
                        layer_args=self.sub_args,
                        reg_args=self.subs_reg_args,
                        norm_args=self.subs_norm_args,
                        is_child=True)

                    _layer_1 = CNN(
                        self.gauge,
                        layer_nm=self.layer_nm + '_cur',
                        layer_args=self.sub_args,
                        reg_args=self.subs_reg_args,
                        norm_args=self.subs_norm_args,
                        is_child=True)

                self.sub_layers.append(_layer_0)
                self.sub_layers.append(_layer_1)

        return

    def call(self, inputs, training, **kwargs):

        if self.tracking_on:
            z_inputs = inputs[0]
            z_tracking = inputs[1]

        if not self.tracking_on:
            z_inputs = inputs
            z_tracking = []

        z_data = z_inputs

        z_sin_pe = None
        z_seq_pe = None
        z_intra_pe = None

        # -- sin pos encoding -- #
        if self.sin_pe_on:
            z_sin_pe = self.gen_sin_pe()
            z_data = z_data + z_sin_pe

        # -- seq pos encoding -- #
        if self.seq_pe_on:
            z_seq_pe = self.gen_seq_pe()
            z_data = z_data + z_seq_pe

        # -- intra-day pos encoding -- #
        if self.intra_pe_on:
            z_intra_pe = self.gen_intra_pe(z_tracking)
            z_data = z_data + z_intra_pe

        z_layers = []

        # -- seq splits -- #
        if self.seq_split:

            # -- loop n_layers for seq splits -- #
            for idx in range(self.n_layers):

                call_layer = self.sub_layers[idx](z_data[:, idx:idx + 1, :], training=training)
                z_layers.append(call_layer)

                if self.seq_scaling_on:
                    z_layers[idx] *= self.scaling_vars[idx]

        # -- no seq splits -- #
        if not self.seq_split:

            if self.n_layers == 1:

                z_layers.append(self.sub_layers[0](z_data, training=training))

            if self.n_layers == 2:

                z_data_0 = z_data[:, :-1, :]
                z_data_1 = z_data[:, -1:, :]

                z_layers.append(self.sub_layers[0](z_data_0, training=training))
                z_layers.append(self.sub_layers[1](z_data_1, training=training))

        # -- all: output_type -- #
        if self.output_type == 'all':

            z_outputs = z_layers

        # -- single: output_type -- #
        if self.output_type == 'single':

            z_outputs = tf.concat(z_layers, axis=1, name='z_outputs')

        # -- split_a: output_type -- #
        if self.output_type == 'split_a':

            past_outputs = tf.concat(z_layers[:-1], axis=1, name='past_outputs')
            cur_outputs = tf.concat(z_layers[-1:], axis=1, name='cur_outputs')

            z_outputs = [past_outputs, cur_outputs]

        # -- split_a: output_type -- #
        if self.output_type == 'split_b':

            all_outputs = tf.concat(z_layers, axis=1, name='all_outputs')
            cur_outputs = tf.concat(z_layers[-1:], axis=1, name='cur_outputs')

            z_outputs = [all_outputs, cur_outputs]

        if self.logs_on:

            _log = {
                'z_inputs': z_inputs,
                'z_tracking': z_tracking,
                'z_sin_pe': z_sin_pe,
                'z_seq_pe': z_seq_pe,
                'z_intra_pe': z_intra_pe,
                'z_data': z_data,
                'z_outputs': z_outputs}

            self.save_layer_log(_log)

        return z_outputs

    def gen_sin_pe(self):

        pos = np.expand_dims(np.arange(0, self.seq_depth), axis=1)
        pe_data = np.zeros((self.seq_depth, self.n_cols))

        div_term = tfm.exp(np.arange(0, self.n_cols, 2) * -(tfm.log(10000.0) / self.n_cols))

        sin_data = np.sin(pos * div_term)
        cos_data = np.cos(pos * div_term)

        pe_data[:, 0::2] = sin_data
        pe_data[:, 1::2] = cos_data

        pe_data = np.repeat(np.expand_dims(pe_data, axis=0), self.batch_size, axis=0)

        return tf.convert_to_tensor(pe_data, name='z_sin_pe')

    def gen_sin_pe_2(self):

        pe_data = np.zeros((self.seq_depth, self.n_cols))
        pos = np.expand_dims(np.arange(self.seq_depth), axis=-1)

        pe_data[0::2] = np.repeat(np.sin(pos),self.n_cols, axis=-1)[0::2]
        pe_data[1::2] = np.repeat(np.cos(pos),self.n_cols, axis=-1)[1::2]


        # pe_data = np.repeat(np.expand_dims(pe_data, axis=0), self.batch_size, axis=0)

        return tf.convert_to_tensor(pe_data, name='z_sin_pe')

    def gen_seq_pe(self):

        pos = np.expand_dims(np.arange(self.seq_depth), axis=-1)

        mm_args = {
            'feature_range': (0,1),
            'copy': True,
            'clip': False}

        mm_trans = getattr(preprocessing, 'MinMaxScaler')(**mm_args)

        pe_data = np.repeat(mm_trans.fit_transform(pos), self.n_cols, axis=-1)

        pe_data = np.repeat(np.expand_dims(pe_data, axis=0), self.batch_size, axis=0)

        return tf.convert_to_tensor(pe_data, name='z_seq_pe')

    def gen_intra_pe(self, tracking):

        t_data = tf.squeeze(tracking[:,:,7:8])

        t_data = (t_data -390)

        mm_args = {
            'feature_range': (0,1),
            'copy': True,
            'clip': False}

        mm_trans = getattr(preprocessing, 'MinMaxScaler')(**mm_args)

        pe_data = np.repeat(np.expand_dims(mm_trans.fit_transform(t_data), -1), self.n_cols, axis=-1)

        return tf.convert_to_tensor(pe_data, name='z_intra_pe')

class DecBars(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, d_model, sin_pe_on, logs_on, **kwargs):
        """
        Layer that produces bars from inputs.

        bars_config:

            n_layers (int 0,1,2):
                0: Create n_layers based on the depth of the seq dim. One layer for each record in the seq_dim
                1: One single layer
                2: Two layers; one for the past records (seq_dim[:-1]) and one layer for the cur records (seq_dim[-1:])

            output_type (string: all, single, split_a, split_b) :
                all: One-to-one output for each internal layer
                single: Single output with all internal layers merged
                split_a: Two outputs; One with all past layers (seq_dim[:-1]) & One with cur layer (seq_dim[-1:])
                split_b: Two outputs; One with all layers & One with cur layer (seq_dim[-1:])

            kernel_size (string: units, cols):
                units: the kernel size and output cols of the internal layers will be set to the units in sub_args
                cols:  the kernel size and output cols of the internal layers will match the num of cols in the last dim of the input

            seq_scale (float) (should set to around .1):
                if !=0: Adds a learnable scaling var for each internal layer. Inits a constant as float(((idx + 1) * seq_scale) + 1)

                constraints: n_layers != 1

        """
        super().__init__(gauge, layer_nm, no_subs=True, **kwargs)

        self.d_model = d_model
        self.sin_pe_on = sin_pe_on
        self.logs_on = logs_on

        self.logs = {}

    def build(self, input_shp):

        self.input_shp = input_shp[0]
        self.batch_size = self.input_shp[0]
        self.seq_depth = self.input_shp[1]
        self.n_cols = self.input_shp[-1]

        self.gauge_dtype = self.gauge.data.dtype

        return

    def call(self, inputs, training, **kwargs):

        z_inputs = inputs[0]
        z_tracking = inputs[1]

        z_data = z_inputs

        z_sin_pe = None
        z_seq_pe = None
        z_intra_pe = None

        z_data = np.repeat(z_data, self.d_model, axis=-1)

        # -- sin pos encoding -- #
        if self.sin_pe_on:
            z_sin_pe = self.gen_sin_pe()
            z_data = z_data + z_sin_pe

        z_data = np.roll(z_data, 1, axis=1)
        z_data[:,0] = 0
        z_outputs = tf.convert_to_tensor(z_data, dtype=self.gauge_dtype)

        if self.logs_on:

            _log = {
                'z_inputs': z_inputs,
                'z_tracking': z_tracking,
                'z_sin_pe': z_sin_pe,
                'z_seq_pe': z_seq_pe,
                'z_intra_pe': z_intra_pe,
                'z_data': z_data,
                'z_outputs': z_outputs}

            self.save_layer_log(_log)

        return z_outputs

    def gen_sin_pe(self):

        pos = np.expand_dims(np.arange(0, self.seq_depth), axis=1)
        pe_data = np.zeros((self.seq_depth, self.d_model))

        div_term = tfm.exp(np.arange(0, self.d_model, 2) * -(tfm.log(10000.0) / self.d_model))

        sin_data = np.sin(pos * div_term)
        cos_data = np.cos(pos * div_term)

        pe_data[:, 0::2] = sin_data
        pe_data[:, 1::2] = cos_data

        pe_data = np.repeat(np.expand_dims(pe_data, axis=0), self.batch_size, axis=0)

        return tf.convert_to_tensor(pe_data, name='z_sin_pe_o')

# -- Transformer -- #

class TauEmbed(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, init_fns, d_model, logs_on=False, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=False, no_log=True, **kwargs)

        self.init_fns = init_fns

        self.d_model = d_model

        self.logs_on = logs_on

        self.logs = []

    def build(self, input_shp):

        self.input_shp = input_shp[0]

        batch_size = self.input_shp[0]
        seq_len = self.input_shp[1]
        n_cols = self.input_shp[-1]

        # trend_k_shp = (seq_len, 1)
        # trend_b_shp = (seq_len, 1)

        # wave_k_shp = (n_cols, self.d_model)
        # wave_b_shp = (seq_len, self.d_model)

        trend_k_shp = (1, 1)
        trend_b_shp = (1, 1)

        wave_k_shp = (1, self.d_model)
        wave_b_shp = (1, self.d_model)

        self.trend_k = self.add_weight(name='trend_k',
                                       shape=trend_k_shp,
                                       initializer=self.init_fns['trend']['kernel'],
                                       trainable=True)

        self.trend_b = self.add_weight(name='trend_b',
                                       shape=trend_b_shp,
                                       initializer=self.init_fns['trend']['bias'],
                                       trainable=True)

        self.wave_k = self.add_weight(name='wave_k',
                                      shape=wave_k_shp,
                                      initializer=self.init_fns['wave']['kernel'],
                                      trainable=True)

        self.wave_b = self.add_weight(name='wave_b',
                                      shape=wave_b_shp,
                                      initializer=self.init_fns['wave']['bias'],
                                      trainable=True)

    def call(self, inputs, training, **kwargs):

        features = inputs[0]

        # -- trend -- #
        trend_data = self.trend_k * features + self.trend_b

        # trend_data = tf.expand_dims(trend_data, axis=-1)

        # -- wave -- #
        wave_dp = tf.tensordot(features, self.wave_k, axes=1) + self.wave_b

        wave_data = tfm.sin(wave_dp)

        z_data = tf.concat([trend_data, wave_data], axis=-1)

        _log = {'features': utils.np_exp(features),
                'trend_k': utils.np_exp(self.trend_k),
                'trend_b': utils.np_exp(self.trend_b),
                'wave_k': utils.np_exp(self.wave_k),
                'wave_b': utils.np_exp(self.wave_b),
                'trend_data': utils.np_exp(trend_data),
                'wave_dp': utils.np_exp(wave_dp),
                'wave_data': utils.np_exp(wave_data),
                'z_data': utils.np_exp(z_data)}

        self.logs.append(_log)

        return z_data

class AttnEmbed(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, dnn_args, emb_type, emb_args, logs_on=False, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=False, no_log=False, **kwargs)

        self.dnn_args = dnn_args

        self.emb_type = emb_type
        self.emb_args = emb_args

        self.d_model = dnn_args['units']

        self.logs_on = logs_on

        self.logs = {}

        self.pos_logs = []

    def build(self, input_shp):

        self.input_shp = input_shp

        self.seq_mask = []

        self.dnn_1 = photon_layers.DNN(self.gauge,
                                       layer_nm='emb_dnn_1',
                                       layer_args=self.dnn_args,
                                       is_child=True)

    def call(self, inputs, training, **kwargs):

        x_inputs = inputs[0]
        x_tracking = inputs[1]

        z_pos = None

        # -- enc inputs -- #
        if self.emb_type == 'enc':
            z_inputs = x_inputs

        # -- dec inputs -- #
        if self.emb_type == 'dec':

            # --- shift decoder inputs down 1 ts -- #
            padding = tf.constant([[
                0,
                0,
            ], [1, 0], [0, 0]])
            z_inputs = tf.pad(x_inputs, padding)[:, :-1]

        z_dnn = self.dnn_1(z_inputs)

        if self.emb_args['pos_encoding'] == 0:
            z_data = z_dnn

        # -- generate pos encoding -- #
        if self.emb_args['pos_encoding'] == 1:
            z_pos = self.gen_pe(z_dnn.shape)
            z_data = z_dnn + z_pos

        if self.emb_args['pos_encoding'] == 2:
            z_pos = self.gen_period_pe(z_dnn.shape, self.emb_args['pe_period'])
            z_data = z_dnn + z_pos

        # -- generate seq_mask -- #
        z_seq_mask = self.gen_seq_mask(x_tracking, dims_out=z_data.shape[-1])

        # -- apply seq mask -- #
        if self.emb_args['mask_on']:
            z_data = z_data.numpy()
            z_data[~z_seq_mask] = 0
            z_data = tf.convert_to_tensor(z_data)

        if self.logs_on:

            _log = {
                'x_inputs': x_inputs,
                'x_tracking': x_tracking,
                'z_inputs': z_inputs,
                'z_dnn': z_dnn,
                'z_pos': z_pos,
                'z_data': z_data,
                'z_seq_mask': z_seq_mask
            }

            self.save_layer_log(_log)

        return z_data

    def gen_pe(self, data_shp):

        batch_size = data_shp[0]
        seq_len = data_shp[1]
        d_model = data_shp[-1]

        pe_data = np.zeros((seq_len, d_model))
        pos = np.expand_dims(np.arange(1,seq_len+1), axis=-1)

        pe_data[0::2] = np.repeat(np.sin(pos),d_model, axis=-1)[0::2]
        pe_data[1::2] = np.repeat(np.cos(pos),d_model, axis=-1)[1::2]

        pe_data = np.repeat(np.expand_dims(pe_data, axis=0), batch_size, axis=0)

        return tf.convert_to_tensor(pe_data, name='z_pos')

    def gen_period_pe(self, data_shp, period):

        batch_size = data_shp[0]
        seq_len = data_shp[1]
        d_model = data_shp[-1]

        pe_data = np.zeros((seq_len, d_model))
        pos = np.expand_dims(np.arange(seq_len), axis=-1)
        pe_data = np.sin(pos * 2 * np.pi / period)
        pe_data = np.repeat(pe_data, d_model, axis=-1)

        pe_data = np.repeat(np.expand_dims(pe_data, axis=0), batch_size, axis=0)

        return pe_data

    def gen_seq_mask(self, x_tracking, dims_out):

        bar_idx = x_tracking[:, :, 0]
        intra_idx = x_tracking[:, :, -1]

        _seq_mask = tf.expand_dims(tf.cast(bar_idx - intra_idx, tf.bool), axis=-1)
        _seq_mask = tf.repeat(_seq_mask, dims_out, axis=-1)

        self.seq_mask.append(_seq_mask)

        return _seq_mask

class Attn(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, layer_args, layer_units, lah_mask_on, logs_on=False, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=False, no_log=False, **kwargs)

        self.layer_args = layer_args
        self.layer_units = layer_units
        self.lah_mask_on = lah_mask_on
        self.logs_on = logs_on

        self.q_layer_nm = self.layer_nm + '_q'
        self.k_layer_nm = self.layer_nm + '_k'
        self.v_layer_nm = self.layer_nm + '_v'

        self.logs = []

    def build(self, input_shp):

        self.q_layer = photon_layers.DNN(self.gauge,
                                         layer_nm=self.q_layer_nm,
                                         layer_args=self.layer_args,
                                         is_child=True)

        self.k_layer = photon_layers.DNN(self.gauge,
                                         layer_nm=self.k_layer_nm,
                                         layer_args=self.layer_args,
                                         is_child=True)

        self.v_layer = photon_layers.DNN(self.gauge,
                                         layer_nm=self.v_layer_nm,
                                         layer_args=self.layer_args,
                                         is_child=True)

        if isinstance(input_shp, list):
            self.seq_len = input_shp[0][1]

        if not isinstance(input_shp, list):
            self.seq_len = input_shp[1]

        self.lah_mask = self.gen_lah_mask(self.seq_len)

    def call(self, inputs, training, **kwargs):

        epoch_idx = utils.np_exp(self.gauge.cur_run.epoch_idx)

        if isinstance(inputs, list):

            q_inputs = inputs[0]
            k_inputs = inputs[1]
            v_inputs = inputs[2]

            q_outputs = self.q_layer(q_inputs, training)
            k_outputs = self.k_layer(k_inputs, training)
            v_outputs = self.v_layer(v_inputs, training)

        if not isinstance(inputs, list):

            q_inputs = inputs
            k_inputs = inputs
            v_inputs = inputs

            q_outputs = self.q_layer(q_inputs, training)
            k_outputs = self.k_layer(k_inputs, training)
            v_outputs = self.v_layer(v_inputs, training)

        attn_outputs, attn_scores = self.scaled_dp(q_outputs, k_outputs, v_outputs)

        if self.logs_on:

            if len(self.logs) <= epoch_idx:
                self.logs.append([])

            self.logs[epoch_idx].append({

                'q_inputs': utils.np_exp(q_inputs),
                'k_inputs': utils.np_exp(k_inputs),
                'v_inputs': utils.np_exp(v_inputs),

                'q_outputs': utils.np_exp(q_outputs),
                'k_outputs': utils.np_exp(k_outputs),
                'v_outputs': utils.np_exp(v_outputs),

                'attn_outputs': utils.np_exp(attn_outputs),
                'attn_scores': utils.np_exp(attn_scores)
            })

        return [attn_outputs, attn_scores[-1]]

    def scaled_dp(self, z_query, z_key, z_value):

        attn_sc_1 = tf.matmul(z_query, z_key, transpose_b=True)
        attn_sc_2 = tf.map_fn(lambda x: x / np.sqrt(self.layer_units), attn_sc_1)

        if self.lah_mask_on:
            attn_sc_3 = attn_sc_2 + (self.lah_mask * -1e9)

        if not self.lah_mask_on:
            attn_sc_3 = attn_sc_2

        attn_sc_4 = tf.nn.softmax(attn_sc_3, axis=1)

        attn_output = tf.matmul(attn_sc_4, z_value)

        return [attn_output, [attn_sc_1,attn_sc_2,attn_sc_3,attn_sc_4]]

    def gen_lah_mask(self, seq_len):
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=np.float64), -1, 0)
        return mask

    def gen_pad_mask(self, seq_data):
        seq = tf.cast(tf.math.equal(seq_data, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

class AttnHeads(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, n_heads, d_model, attn_args, split_heads, lah_mask_on, log_scores, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=False, no_log=False, **kwargs)

        self.n_heads = n_heads
        self.d_model = d_model
        self.attn_args = attn_args
        self.split_heads = split_heads
        self.lah_mask_on = lah_mask_on
        self.log_scores = log_scores

        self.layer_units = self.d_model

        self.head_axis = 1

        if not self.split_heads:
            self.head_axis = -1

        self.attn_heads = []
        self.attn_scores = []

    def build(self, input_shp):

        self.dnn_args = {
            'units': self.d_model,
            'activation': self.attn_args['act_fn'],
            'kernel_initializer': self.attn_args['k_init'],
            'use_bias': self.attn_args['use_bias'],
            'bias_initializer': self.attn_args['b_init'],
            'kernel_regularizer': self.attn_args['w_reg'],
            'bias_regularizer': self.attn_args['b_reg'],
            'activity_regularizer': self.attn_args['a_reg'],
            'trainable': True}

        self.heads_out = photon_layers.DNN(self.gauge,
                                           layer_nm=self.layer_nm + '_out',
                                           layer_args=self.dnn_args,
                                           is_child=True)

        self.layer_args = self.dnn_args
        self.layer_args['units'] = self.layer_units

        # --- create attn heads --- #
        for h in range(self.n_heads):

            layer_nm = self.layer_nm + '_' + str(h + 1)

            self.attn_heads.append(
                Attn(gauge=self.gauge,
                     layer_nm=layer_nm,
                     layer_args=self.layer_args,
                     layer_units=self.layer_units,
                     lah_mask_on=self.lah_mask_on,
                     logs_on=self.attn_args['logs_on'],
                     is_child=True))

    def call(self, inputs, training, **kwargs):

        # -- split inputs -- #
        if self.split_heads:

            if isinstance(inputs, list):

                q_inputs = tf.split(inputs[0], self.n_heads, axis=1)
                k_inputs = tf.split(inputs[1], self.n_heads, axis=1)
                v_inputs = tf.split(inputs[2], self.n_heads, axis=1)

                split_inputs = []
                for h in range(self.n_heads):
                    split_inputs.append([q_inputs[h], k_inputs[h], v_inputs[h]])

            else:
                split_inputs = tf.split(inputs, self.n_heads, axis=1)

        attn_outputs = []

        # -- loop attn heads -- #
        for h in range(self.n_heads):

            # -- split head inputs --#
            if self.split_heads:

                z_heads = self.attn_heads[h](split_inputs[h], training)

            else:

                z_heads = self.attn_heads[h](inputs, training)

            # -- save attn outputs -- #
            attn_outputs.append(z_heads[0])

            # -- log attn scores -- #
            if self.log_scores:

                # -- place for head scores -- #
                if len(self.attn_scores) <= h:
                    self.attn_scores.append([])

                # -- save scores -- #
                self.attn_scores[h].append(z_heads[1])

        # -- merge outputs -- #
        z_output = tf.concat(attn_outputs, axis=self.head_axis)

        return self.heads_out(z_output, training)

class EncoderBlock(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, n_heads, d_model, attn_args, split_heads, lah_mask_on, log_scores, dff, res_config, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=False, no_log=True, **kwargs)

        self.n_heads = n_heads
        self.d_model = d_model
        self.attn_args = attn_args
        self.split_heads = split_heads
        self.lah_mask_on = lah_mask_on
        self.log_scores = log_scores
        self.dff = dff

        self.res_config = res_config

        self.sah_nm = self.layer_nm + '_sah'
        self.res_1_nm = self.layer_nm + '_res_1'
        self.res_2_nm = self.layer_nm + '_res_2'
        self.ffn_1_nm = self.layer_nm + '_ffn_1'
        self.ffn_2_nm = self.layer_nm + '_ffn_2'

        # -- self attn heads -- #
        self.sah_args = {
            'n_heads': self.n_heads,
            'd_model': self.d_model,
            'attn_args': self.attn_args,
            'split_heads': self.split_heads,
            'lah_mask_on': self.lah_mask_on,
            'log_scores': self.log_scores
        }

        self.ffn_1_args = {
            'units': self.dff,
            'activation': 'relu',
            'kernel_initializer': self.attn_args['k_init'],
            'use_bias': self.attn_args['use_bias'],
            'bias_initializer': self.attn_args['b_init'],
            'trainable': True
        }

        self.ffn_2_args = {
            'units': self.d_model,
            'activation': None,
            'kernel_initializer': self.attn_args['k_init'],
            'use_bias': self.attn_args['use_bias'],
            'bias_initializer': self.attn_args['b_init'],
            'trainable': True
        }

    def build(self, input_shp):

        # -- self attn heads -- #
        self.sah = AttnHeads(gauge=self.gauge, layer_nm=self.sah_nm, is_child=True, **self.sah_args)

        # -- add first res layer -- #
        self.res_1 = Res(gauge=self.gauge, layer_nm=self.res_1_nm, res_config=self.res_config, is_child=True)

        # -- add first ffn layer -- #
        self.ffn_1 = photon_layers.DNN(gauge=self.gauge,
                                       layer_nm=self.ffn_1_nm,
                                       layer_args=self.ffn_1_args,
                                       is_child=True)

        # -- add second ffn layer -- #
        self.ffn_2 = photon_layers.DNN(gauge=self.gauge,
                                       layer_nm=self.ffn_2_nm,
                                       layer_args=self.ffn_2_args,
                                       is_child=True)

        # -- add second res layer -- #
        self.res_2 = photon_layers.Res(gauge=self.gauge,
                                       layer_nm=self.res_2_nm,
                                       res_config=self.res_config,
                                       is_child=True)

    def call(self, inputs, training, **kwargs):

        z_sah = self.sah(inputs, training)

        z_res_1 = self.res_1([inputs, z_sah], training)

        z_ffn_1 = self.ffn_1(z_res_1, training)
        z_ffn_2 = self.ffn_2(z_ffn_1, training)

        z_res_2 = self.res_1([z_res_1, z_ffn_2], training)

        return z_res_2

class DecoderBlock(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, n_heads, d_model, attn_args, split_heads, lah_mask_on, log_scores, dff, res_config, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=False, no_log=False, **kwargs)

        self.n_heads = n_heads
        self.d_model = d_model
        self.attn_args = attn_args
        self.split_heads = split_heads
        self.lah_mask_on = lah_mask_on
        self.log_scores = log_scores
        self.dff = dff

        self.res_config = res_config

        self.sah_nm = self.layer_nm + '_sah'
        self.edah_nm = self.layer_nm + '_edah'
        self.res_1_nm = self.layer_nm + '_res_1'
        self.res_2_nm = self.layer_nm + '_res_2'
        self.res_3_nm = self.layer_nm + '_res_3'
        self.ffn_1_nm = self.layer_nm + '_ffn_1'
        self.ffn_2_nm = self.layer_nm + '_ffn_2'

        # -- self attn heads -- #
        self.sah_args = {
            'n_heads': self.n_heads,
            'd_model': self.d_model,
            'attn_args': self.attn_args,
            'split_heads': self.split_heads,
            'lah_mask_on': self.lah_mask_on,
            'log_scores': self.log_scores
        }

        self.edah_args = {
            'n_heads': self.n_heads,
            'd_model': self.d_model,
            'attn_args': self.attn_args,
            'split_heads': self.split_heads,
            'lah_mask_on': self.lah_mask_on,
            'log_scores': self.log_scores
        }

        self.ffn_1_args = {
            'units': self.dff,
            'activation': 'relu',
            'kernel_initializer': self.attn_args['k_init'],
            'use_bias': self.attn_args['use_bias'],
            'bias_initializer': self.attn_args['b_init'],
            'trainable': True
        }

        self.ffn_2_args = {
            'units': self.d_model,
            'activation': None,
            'kernel_initializer': self.attn_args['k_init'],
            'use_bias': self.attn_args['use_bias'],
            'bias_initializer': self.attn_args['b_init'],
            'trainable': True
        }

    def build(self, input_shp):

        # -- add first res layer -- #
        self.res_1 = Res(gauge=self.gauge, layer_nm=self.res_1_nm, res_config=self.res_config, is_child=True)

        # -- self attn heads -- #
        self.sah = AttnHeads(gauge=self.gauge, layer_nm=self.sah_nm, **self.sah_args)

        # -- enc/dec attn heads -- #
        self.edah = AttnHeads(gauge=self.gauge, layer_nm=self.edah_nm, **self.edah_args)

        # -- add second res layer -- #
        self.res_2 = Res(gauge=self.gauge, layer_nm=self.res_2_nm, res_config=self.res_config, is_child=True)

        # -- add second res layer -- #
        self.res_3 = Res(gauge=self.gauge, layer_nm=self.res_3_nm, res_config=self.res_config, is_child=True)

        # -- add first ffn layer -- #
        self.ffn_1 = photon_layers.DNN(gauge=self.gauge,
                                       layer_nm=self.ffn_1_nm,
                                       layer_args=self.ffn_1_args)

        # -- add second ffn layer -- #
        self.ffn_2 = photon_layers.DNN(gauge=self.gauge,
                                       layer_nm=self.ffn_2_nm,
                                       layer_args=self.ffn_2_args)

    def call(self, inputs, training, **kwargs):

        dec_inputs = inputs[0]
        enc_outputs = inputs[1]

        z_sah = self.sah(dec_inputs, training)
        z_res_1 = self.res_1([dec_inputs, z_sah], training)

        z_edah = self.edah([enc_outputs, enc_outputs, z_res_1], training)

        z_res_2 = self.res_2([enc_outputs, z_edah], training)

        z_ffn_1 = self.ffn_1(z_edah, training)
        z_ffn_2 = self.ffn_2(z_ffn_1, training)

        z_res_3 = self.res_3([z_ffn_2, z_res_2], training)

        return z_res_3

class EncoderStack(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, n_blocks, block_args, logs_on=False, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=False, no_log=False, **kwargs)

        self.n_blocks = n_blocks
        self.block_args = block_args
        self.logs_on = logs_on

        self.logs = {}

    def build(self, input_shp):

        self.enc_blocks = []

        # -- loop blocks -- #
        for b in range(self.n_blocks):

            layer_nm = self.layer_nm + '_blk_' + str(b + 1)

            # -- add encoder blocks -- #
            self.enc_blocks.append(EncoderBlock(gauge=self.gauge, layer_nm=layer_nm, is_child=True, **self.block_args))

    def call(self, inputs, training, **kwargs):

        z_enc_blocks = []

        z_inputs = inputs

        # -- loop blocks -- #
        for b in range(self.n_blocks):

            if b > 0:
                z_inputs = z_enc_blocks[-1]

            # -- call encoder blocks -- #
            z_enc_blocks.append(self.enc_blocks[b](z_inputs, training))

        if self.logs_on:

            _log = {'z_inputs': z_inputs, 'z_enc_blocks': z_enc_blocks}

            self.save_layer_log(_log)

        return z_enc_blocks[-1]

class DecoderStack(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, n_blocks, block_args, logs_on=False, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=False, no_log=False, **kwargs)

        self.n_blocks = n_blocks
        self.block_args = block_args
        self.logs_on = logs_on

        self.logs = {}

    def build(self, input_shp):

        self.dec_blocks = []

        # -- loop blocks -- #
        for b in range(self.n_blocks):

            layer_nm = self.layer_nm + '_' + str(b + 1)

            # -- add encoder blocks -- #
            self.dec_blocks.append(DecoderBlock(gauge=self.gauge, layer_nm=layer_nm, is_child=True, **self.block_args))

    def call(self, inputs, training, **kwargs):

        dec_inputs = inputs[0]
        enc_outputs = inputs[1]

        block_inputs = [dec_inputs, dec_inputs, enc_outputs]

        z_dec_blocks = []

        # -- loop blocks -- #
        for b in range(self.n_blocks):

            if b > 0:
                block_inputs = [z_dec_blocks[-1], enc_outputs]

            # -- call encoder blocks -- #
            z_dec_blocks.append(self.dec_blocks[b](block_inputs, training))

        if self.logs_on:

            _log = {
                'dec_inputs': dec_inputs,
                'enc_outputs': enc_outputs,
                'block_inputs': block_inputs,
                'z_dec_blocks': z_dec_blocks
            }

            self.save_layer_log(_log)

        return z_dec_blocks[-1]

class TransOut(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, dnn_args, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=False, no_log=False, **kwargs)

        self.dnn_args = dnn_args

    def build(self, input_shp):

        self.flatten = tf_layers.Flatten()

        self.filter = photon_layers.DNN(self.gauge,
                                        layer_nm=self.layer_nm + '_filter',
                                        layer_args=self.dnn_args)

    def call(self, inputs, training, **kwargs):

        z_flatten = self.flatten(inputs)

        z_filter = self.filter(z_flatten)

        return tf.matmul(z_filter, inputs)

class TimeDis(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, dis_layer, **kwargs):

        super().__init__(gauge, layer_nm, **kwargs)

        self.dis_layer = dis_layer

    def build(self, input_shp):

        self.input_shp = input_shp

        self.k_layer = tf_layers.TimeDistributed(name=self.layer_nm, layer=self.dis_layer)

        return

    def call(self, inputs, training, **kwargs):

        self.z_output = self.k_layer(inputs, training=training)

        return self.z_output

class PosEnc(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, emb_type, d_model, w_fns, logs_on=False, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=False, no_log=True, **kwargs)

        self.emb_type = emb_type
        self.d_model = d_model
        self.w_fns = w_fns

        self.logs_on = logs_on

        self.logs = []

    def build(self, input_shp):

        self.input_shp = input_shp

        self.trend_k = self.add_weight(name=self.layer_nm + '/trend_k',
                                       shape=(self.input_shp[1], ),
                                       initializer=self.w_fns['trend']['k_init'],
                                       regularizer=self.w_fns['trend']['k_reg'],
                                       trainable=True)

        self.trend_b = self.add_weight(name=self.layer_nm + '/trend_b',
                                       shape=(self.input_shp[1], ),
                                       initializer=self.w_fns['trend']['b_init'],
                                       regularizer=self.w_fns['trend']['b_reg'],
                                       trainable=True)

        self.sin_k = self.add_weight(name=self.layer_nm + '/sin_k',
                                     shape=(self.input_shp[1], self.d_model),
                                     initializer=self.w_fns['sin']['k_init'],
                                     regularizer=self.w_fns['sin']['k_reg'],
                                     trainable=True)

        self.sin_b = self.add_weight(name=self.layer_nm + '/sin_b',
                                     shape=(self.input_shp[1], self.input_shp[1], self.d_model),
                                     initializer=self.w_fns['sin']['b_init'],
                                     regularizer=self.w_fns['sin']['b_reg'],
                                     trainable=True)

    def call(self, inputs, training, **kwargs):

        self.features = inputs

        self.trend_data = self.trend_k * self.features

        # + self.trend_b

        self.sin_dp = tf.matmul(self.features, self.sin_k)

        # + self.sin_b

        self.sin_data = tfm.sin(self.sin_dp)

        # self.z_merge = tf.concat([tf.expand_dims(self.trend_data, -1), self.sin_data], axis=-1)

        # self.z_data = tf.reshape(self.z_merge, shape=(-1, self.seq_len*(self.d_model+1)))

        _log = {'features': self.features,
                'trend_data': self.trend_data,
                'sin_dp': self.sin_dp,
                'sin_data': self.sin_data}
                # 'z_merge': self.z_merg}
                # 'z_data': self.z_data}

        self.logs.append(_log)

        return self.features

class DynamicEnc(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, sub_args, logs_on=False, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=False, no_log=True, **kwargs)

        self.sub_args = sub_args
        self.logs_on = logs_on

        self.sub_layers = []

        self.logs = []

    def build(self, input_shp):

        self.input_shp = input_shp
        self.n_inputs = 1

        if isinstance(self.input_shp, list):
            self.n_inputs = len(self.input_shp)

        for idx in range(self.n_inputs):

            nm = self.layer_nm + '_sub_' + str(idx)

            _layer = RNN(self.gauge,
                        layer_nm=nm,
                        rnn_type='lstm',
                        rnn_args=self.sub_args,
                        is_child=True)

            self.sub_layers.append(_layer)

    def call(self, inputs, training, **kwargs):

        z_layers = []

        # -- loop n_layers for seq splits -- #
        for idx in range(self.n_inputs):

            if self.n_inputs == 1:
                layer_inputs = inputs

            if self.n_inputs > 1:
                layer_inputs = inputs[idx]

            call_layer = self.sub_layers[idx](layer_inputs, training=training)
            z_layers.append(call_layer)

        return z_layers[0]
