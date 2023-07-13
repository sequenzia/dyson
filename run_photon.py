import sys
import os
from dyson import config
from photon import Photon
from photon import plot_loss

import pandas as pd
from matplotlib import rcParams
import plotly.graph_objects as go
import plotly.express as px

FRAMEWORK = 'tf'
RUN_DIR = '/var/lib/alpha/omega/.run'
DATA_DIR = '/var/lib/alpha/omega/dyson/dyson/data'

config.network_config['data_dir'] = DATA_DIR
config.network_config['data_fn'] = 'SPY_1T_2016_2017'

photon = Photon(FRAMEWORK, RUN_DIR)

net = photon.Networks(photon=photon, **config.network_config)
tree = photon.Trees(network=net, **config.tree_config)

# branch = photon.Branches(trees=[tree], **config.dnn_config)
# branch = photon.Branches(trees=[tree], **config.cnn_config)
# branch = photon.Branches(trees=[tree], **config.ens_config)
# branch = photon.Branches(trees=[tree], **config.rnn_config)
branch = photon.Branches(trees=[tree], **config.trans_config)

run = net.gamma.run_network(branches=[branch])

# loss_data = run.run_data[0]

# plot_loss(loss_data)

