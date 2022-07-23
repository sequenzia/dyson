run_branch = run.branches[0]
src_branch = run_branch.src

run_chain = run_branch.chains[0]
src_chain = run_chain.src

run_model = run_chain.models[0]

tree = src_branch.trees[0]

gauge = src_chain.models[0]

model = gauge.src


layer_logs = run.branches[0].chains[0].src.models[0].logs.layers['main'][0][1]

batch_data_logs = trans_branch.chains[0].logs.batch_data['main']

for x in layer_logs:
    print(x['layer_name'], x['in_shape'], x['out_shape'])


layer_logs[0]['input'].shape


batch_data_logs[0][0]['tracking'][0]


gauge.layers['dec_bars'].logs[0][0].keys()

gauge.layers['dec_bars'].logs[0][0]['z_data']