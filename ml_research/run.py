import sys, os
import tensorflow as tf
import numpy as np

run_local = False
local_path = '/var/lib/alpha/photon/pkg/src/photon/'
cuda_devices = [0,1,2]

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(e) for e in cuda_devices)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if run_local:
    sys.path.append(local_path)

from photon import Photon
from configs import ensemble_config as config

photon = Photon(run_local=run_local)

neon = photon.Networks(photon=photon, **config.neon_config)
argon = photon.Trees(network=neon, **config.argon_config)
muon = photon.Branches(trees=[argon], **config.muon_config)

run = neon.gamma.run_network(branches=[muon])
