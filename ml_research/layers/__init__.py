from photon import layers as photon_layers
from tensorflow.keras import layers as tf_layers

class Dense(photon_layers.Layers):

    def __init__(self, gauge, layer_nm, layer_args, **kwargs):

        super().__init__(gauge, layer_nm, **kwargs)

        self.layer_args = layer_args

    def build(self, input_shp):

        self.input_shp = input_shp

        self.k_layer = tf_layers.Dense(name=self.layer_nm, **self.layer_args)

        return

    def call(self, inputs, training=None, **kwargs):
        print('call users ddnn')
        return self.k_layer(inputs=inputs, training=training)