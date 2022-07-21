run_branch = run.branches[0]
src_branch = run_branch.src

run_chain = run_branch.chains[0]
src_chain = run_chain.src

run_model = run_chain.models[0]

gauge = src_chain.models[0]

model = gauge.src


layer_logs = run.branches[0].chains[0].src.models[0].logs.layers['main']

# # --- log calls --- #
# if self.gauge.is_model_built and self.log_calls and not self.gauge.run_model.live.is_val:
#
#     if len(self.self.call_logs['main']) <= epoch_idx:
#         self.self.call_logs['main'].append([])
#
#     _log = {'layers': log_data, 'z_outputs': z_outputs}
#
#     self.rd_logs[epoch_idx].insert(batch_idx, _log)
#

