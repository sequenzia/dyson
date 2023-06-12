import sys, os
import tensorflow as tf
import numpy as np

run_local = True
local_path = '/var/lib/alpha/omega/photon/src'
data_dir = '/var/lib/alpha/omega/dyson/ml_research/data'
cuda_devices = [0,]

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(e) for e in cuda_devices)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from photon import Photon

import config

config.network_config['data_dir'] = data_dir
config.network_config['data_fn'] = 'SPY_1T_2016_2017'

photon = Photon(run_local=run_local)

net = photon.Networks(photon=photon, **config.network_config)
tree = photon.Trees(network=net, **config.tree_config)

cnn_branch = photon.Branches(trees=[tree], **config.cnn_config)
ens_branch = photon.Branches(trees=[tree], **config.ens_config)
rnn_branch = photon.Branches(trees=[tree], **config.rnn_config)
trans_branch = photon.Branches(trees=[tree], **config.trans_config)

run = net.gamma.run_network(branches=[trans_branch])

