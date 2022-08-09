run_branch = run.branches[0]
src_branch = run_branch.src

run_chain = run_branch.chains[0]
src_chain = run_chain.src

run_model = run_chain.models[0]

tree = src_branch.trees[0]

gauge = src_chain.models[0]

run_chain.specs['run_spec']

gauge.logs.theta[0][0]['model_pre'][0]['avg']
gauge.logs.theta[0][0]['grads'][0]['avg']
gauge.logs.theta[0][0]['model_post'][0]['avg']

gauge.logs.theta[0][0]['opt'][1]


model = gauge.src

model.log_theta


layer_logs = run.branches[0].chains[0].src.models[0].logs.layers['main'][0][1]

batch_data_logs = trans_branch.chains[0].logs.batch_data['main']

for x in layer_logs:
    print(x['layer_name'], x['in_shape'], x['out_shape'])


layer_logs[0]['input'].shape


batch_data_logs[0][0]['tracking'][0]


gauge.layers['dec_bars'].logs[0][0].keys()

gauge.layers['dec_bars'].logs[0][0]['z_data']


[[{'async_idx': 0, 'device_idx': 0, 'max_wkrs': 80,
   '_start_idx': 0, '_end_idx': 80, 'start_idx': 0,
   'end_idx': 1}, {'async_idx': 0, 'device_idx': 1,
                   'max_wkrs': 10, '_start_idx': 80, '_end_idx': 90,
                   'start_idx': None, 'end_idx': None},
  {'async_idx': 0, 'device_idx': 2, 'max_wkrs': 10, '_start_idx': 90,
   '_end_idx': 100, 'start_idx': None, 'end_idx': None}]]
