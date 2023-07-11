from dyson import config
from photon import Photon
from typing import Any, List
from dataclasses import dataclass, field, replace as dc_replace
import tensorflow as tf

from dyson.models.cnn_models import CNN_Base
from photon.runs import setup_runs


FRAMEWORK = 'tf'
RUN_DIR = '/var/lib/alpha/omega/.run'
DATA_DIR = '/var/lib/alpha/omega/dyson/dyson/data'

config.network_config['data_dir'] = DATA_DIR
config.network_config['data_fn'] = 'SPY_1T_2016_2017'


def get_data(chain):

    data_type = chain.live.data_type

    dataset = chain.src.datasets[chain.live.tree_idx][data_type]

    for batch_idx, data in enumerate(dataset):

        inputs = data[0]
        targets = data[1]
        tracking = data[2]
        outputs = data[3]

        if not chain.src.data_config['targets']['is_seq'] and chain.src.trees[0].data.seq_on:
            targets = data[1][..., -1, :]

        batch_data = {'inputs': inputs,
                        'targets': targets,
                        'tracking': tracking,
                        'outputs': outputs}

        yield batch_idx, batch_data


def run_splits(model, step_data, batch_data):

    targets_config = model.chain.src.data_config['targets']

    if step_data['y_true'] is None:
        step_data['y_true'] = batch_data['targets'][targets_config['true_slice']]

    if step_data['y_tracking'] is None:
        step_data['y_tracking'] = batch_data['targets'][targets_config['tracking_slice']]

    return step_data


def run_loss(model, step_data):

    # -- step loss -- #
    step_data['step_loss'] = model.gauge.loss_fn(step_data['y_true'], step_data['y_hat'])

    if tf.rank(step_data['step_loss']) > 0:
        step_data['step_loss'] = tf.nn.compute_average_loss(step_data['step_loss'],
                                                            global_batch_size=model.chain.branch.src.trees[model.chain.live.tree_idx].data.batch_size)
    # -- model loss -- #
    step_data['model_loss'] = sum(model.src.losses)

    # -- full loss -- #
    step_data['full_loss'] = tf.add(step_data['step_loss'], step_data['model_loss'])

    return step_data



def run_model(batch_idx, batch_data, model):

    model.gauge.is_live = True

    model.live.is_training = True

    # --- load tape --- #
    with tf.GradientTape() as tape:

        with tf.device('/GPU:0'):

            # -- run model -- #
            step_data = model.src(inputs=batch_data['inputs'],
                                    training=True,
                                    batch_idx=batch_idx,
                                    targets=batch_data['targets'],
                                    tracking=batch_data['tracking'])

            step_data = run_splits(model, step_data, batch_data)

            step_data = run_loss(model, step_data)

            # -- run grads -- #
            step_data['step_grads'] = tape.gradient(step_data['step_loss'],
                                                    model.src.trainable_variables)

            # -- apply grads -- #
            model.src.optimizer.apply_grads(list(zip(step_data['step_grads'],
                                                     model.src.trainable_variables)))

    return step_data


def run_batches(epoch_idx, chain, model_idx):

    step_data = []

    for batch_idx, batch_data in get_data(chain):

        print(f"{epoch_idx} -> {batch_idx}")

        model_step_data = run_model(batch_idx, batch_data, chain.models[model_idx])

        step_data.append(model_step_data)

    return step_data


def run_epochs(n_epochs, chain, model_idx):

    run_data = []
    for epoch_idx in range(n_epochs):
        run_data.append(run_batches(epoch_idx, chain, model_idx))

    return run_data




photon = Photon(FRAMEWORK, RUN_DIR)

network = photon.Networks(photon=photon, **config.network_config)
tree = photon.Trees(network=network, **config.tree_config)
branch = photon.Branches(trees=[tree], **config.cnn_config)

tree_idx = tree.tree_idx
branch_idx = branch.branch_idx
chain_idx = 0
model_idx = 0

run = setup_runs(network, [branch], [], None, None)


chain = run.branches[branch_idx].chains[chain_idx]

chain.live.data_type = 'train'
chain.live.tree_idx = tree_idx

run_data = run_epochs(200, chain, model_idx)










# branch = run.branches[branch_idx].src
# branch_configs = branch.configs
# model_config = branch_configs.model_config[tree_idx]
# opt_config = branch_configs.opt_config[tree_idx]

# run_branch = run.branches[branch_idx]
# run_branch.add_chain(run_branch.src.chains[chain_idx], chain_idx)

# run_chain = run_branch.chains[chain_idx]
# run_chain.add_live()
# run_chain.live.data_type = 'train'
# run_chain.live.tree_idx = tree_idx
# run_chain.src.opt_config = opt_config

# gauge = run_branch.src.chains[chain_idx].models[model_idx]


# run_chain.add_model(gauge, model_idx)
# run_chain.models[model_idx].add_live()


# run_chain.models[model_idx].gauge.model_args = model_config['args']
# run_chain.models[model_idx].gauge.opt_fn = opt_config['fn']


# run_chain.models[model_idx].gauge.build_gauge(run, run_chain, run_chain.models[model_idx])






# run_chain.models[model_idx].gauge.compile_gauge(tree_idx)


# run_chain.models[model_idx].gauge.pre_build_model()







# epoch_idx = 0

# run_model = run_chain.models[model_idx]

# for batch_idx, batch_data in get_data(run_chain):

#     print(f"{epoch_idx} -> {batch_idx}")

#     run_chain.live.batch_idx = batch_idx
#     run_model.live.batch_idx = batch_idx



#     run_model.src(batch_data['inputs'], training=True)





# run_chain.models[model_idx].gauge.pre_build_model()


# args = {'gauge': run_chain.models[model_idx].run_gauge}

# run_chain.models[model_idx].src = model_config['model'](**args)

# run_chain.models[model_idx].src.pre_build()

# run_chain.models[model_idx].src.build_model()

# run_chain.models[model_idx].gauge.run_model = run_chain.models[model_idx]



