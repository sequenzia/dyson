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

# --- Batch Logs --- #

batch_logs = chain.logs.batch_data['main']

batch_logs[0][0].keys()

inputs = batch_logs[0][0]['inputs']
targets = batch_logs[0][0]['targets']
tracking = batch_logs[0][0]['tracking']
outputs = batch_logs[0][0]['outputs']

import tensorflow as tf

outputs[:,-1,:]

inputs[0][:25]

outputs[0][:25]


# --- Call Logs --- #

# --- Layer Logs --- #
"""
    gauge.logs.layers[log_type][epoch_idx][batch_idx][layer_idx]

    keys = ['step' : [chain_idx, epoch_idx, batch_idx],
            'layer',
            'layer_idx',
            'layer_name',
            'in_shape',
            'out_shape',
            'input',
            'output',
            'loss']

    for x in layer_logs:
        print(x['layer_name'], x['in_shape'], x['out_shape'])

"""

layer_logs = gauge.logs.layers['main'][0][0][2]

print(layer_logs['layer_name'])
print(layer_logs['in_shape'])
print(layer_logs['out_shape'])



# --- Run Data Logs -- #
""" """

# ---  Theta Logs --- #
"""
    gauge.logs.theta[epoch_idx][batch_idx]

    keys = ['model_pre', 'model_post', 'opt', 'grads']

    tf.subtract(pre,tf.math.multiply(lr,grads))

    tf.math.reduce_sum([pre, - tf.math.mulgrads])

    tf.math.reduce_sum(post)

"""

pre_layers = gauge.logs.theta[0][0]['model_pre']

for p in pre_layers:
    print(p['name'])


gauge.logs.theta[0][0]['model_pre'][0]['value'][0]
gauge.logs.theta[0][0]['model_pre'][3]['value'][0]

pre = gauge.logs.theta[0][0]['model_pre'][0]['value'][0]

grads = gauge.logs.theta[0][0]['grads'][0]['value'][0]
post = gauge.logs.theta[0][0]['model_post'][0]['value'][0]
lr = gauge.logs.theta[0][0]['opt'][0]['value'][0]

gauge.layers['dnn_1'].trainable_variables[0]

gauge.layers['dnn_1'].k_layer.trainable_weights[0]