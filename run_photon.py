import sys, os
from dyson import config
from photon import Photon
import tensorflow as tf
import torch
from torch.utils.data import DataLoader

FRAMEWORK = 'tf'
RUN_DIR = '/var/lib/alpha/omega/.run'
DATA_DIR = '/var/lib/alpha/omega/dyson/dyson/data'

config.network_config['data_dir'] = DATA_DIR
config.network_config['data_fn'] = 'SPY_1T_2016_2017'

photon = Photon(FRAMEWORK, RUN_DIR)

net = photon.Networks(photon=photon, **config.network_config)
tree = photon.Trees(network=net, **config.tree_config)

# branch = photon.Branches(trees=[tree], **config.dnn_config)
branch = photon.Branches(trees=[tree], **config.cnn_config)
# branch = photon.Branches(trees=[tree], **config.ens_config)
# branch = photon.Branches(trees=[tree], **config.rnn_config)
# branch = photon.Branches(trees=[tree], **config.trans_config)

run = net.gamma.run_network(branches=[branch])

chain = branch.chains[0]
model = chain.models[0]

# train_dataloader = DataLoader(tree.datasets['train'], batch_size=64, shuffle=False)

# dir(run.branches[0].chains[0].models[0].steps)

# run.branches[0].chains[0].models[0].steps.y_true[0]

