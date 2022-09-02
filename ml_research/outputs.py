tree_idx = 0
branch_idx = 0
chain_idx = 0
model_idx = 0

# ------- Setup Structure ------- #

run_branch = run.branches[branch_idx]
branch = run_branch.src

tree = branch.trees[tree_idx]

run_chain = run_branch.chains[chain_idx]
chain = run_chain.src

run_model = run_chain.models[model_idx]
gauge = chain.models[model_idx]

gauge.layers['cnn_1'].k_layer.losses

# --- Steps --- #
"""
['features',
 'full_loss',
 'grads',
 'learning_rates',
 'metrics',
 'model',
 'model_loss',
 'preds_dist',
 'preds_samples',
 'specs',
 'step_loss',
 'x_tracking',
 'y_hat',
 'y_tracking',
 'y_true']
"""

import tensorflow as tf, numpy as np

epoch_idx = 0
batch_idx = 0

steps = run_model.steps

y_true = steps.y_true[epoch_idx][batch_idx][75:]
y_hat = steps.y_hat[epoch_idx][batch_idx][75:]

threshold = .35

TruePositives = tf.keras.metrics.TruePositives(threshold)
FalsePositives = tf.keras.metrics.FalsePositives(threshold)


TruePositives.update_state(y_true,y_hat)
FalsePositives.update_state(y_true,y_hat)

TruePositives.result()
FalsePositives.result()



Precision = tf.keras.metrics.Precision(threshold)

Precision.update_state(y_true,y_hat)

Precision.result()