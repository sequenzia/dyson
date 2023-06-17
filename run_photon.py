import sys, os
import numpy as np, tensorflow as tf
import config
from photon import Photon

RUN_LOCAL = True
LOCAL_PATH = '/var/lib/alpha/omega/photon/src'
DATA_DIR = '/var/lib/alpha/omega/dyson/ml_research/data'

config.network_config['data_dir'] = DATA_DIR
config.network_config['data_fn'] = 'SPY_1T_2016_2017'

photon = Photon(run_local=RUN_LOCAL)

net = photon.Networks(photon=photon, **config.network_config)
tree = photon.Trees(network=net, **config.tree_config)

cnn_branch = photon.Branches(trees=[tree], **config.cnn_config)
ens_branch = photon.Branches(trees=[tree], **config.ens_config)
rnn_branch = photon.Branches(trees=[tree], **config.rnn_config)
trans_branch = photon.Branches(trees=[tree], **config.trans_config)

run = net.gamma.run_network(branches=[trans_branch])

