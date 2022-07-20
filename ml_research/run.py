import sys, os
import tensorflow as tf
import numpy as np

run_local = True
local_path = '/var/lib/alpha/photon/pkg/src/'
cuda_devices = [0,1,2]

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(e) for e in cuda_devices)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if run_local:
    sys.path.insert(0,local_path)

import photon, config

photon = photon.Photon(run_local=run_local)

net = photon.Networks(photon=photon, **config.network_config)
tree = photon.Trees(network=net, **config.tree_config)

en_branch = photon.Branches(trees=[tree], **config.ens_config)
cnn_branch = photon.Branches(trees=[tree], **config.cnn_config)
rnn_branch = photon.Branches(trees=[tree], **config.rnn_config)
trans_branch = photon.Branches(trees=[tree], **config.trans_config)

run = net.gamma.run_network(branches=[trans_branch])

