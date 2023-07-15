from photon import Photon, Networks, Trees, Branches
from dyson import config

FRAMEWORK = 'tf'
RUN_DIR = '/var/lib/alpha/omega/.run'
DATA_DIR = '/var/lib/alpha/omega/dyson/dyson/data'

config.network_config['data_dir'] = DATA_DIR
config.network_config['data_fn'] = 'SPY_1T_2016_2017'

photon = Photon(FRAMEWORK, RUN_DIR)

network = Networks(photon=photon, **config.network_config)
tree = Trees(network=network, **config.tree_config)
branch = Branches(trees=[tree], **config.dnn_config)
# branch = Branches(trees=[tree], **config.cnn_config)
# branch = Branches(trees=[tree], **config.ens_config)
# branch = Branches(trees=[tree], **config.rnn_config)
# branch = Branches(trees=[tree], **config.trans_config)

run = network.gamma.run_network(branches=[branch])
