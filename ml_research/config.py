from photon import metrics as photon_metrics, losses, optimizers, utils, options
from models import ens_models, cnn_models, rnn_models, trans_models, prob_models
import metrics

metrics = metrics.Metrics()

losses = losses.Losses()
photon_metrics = photon_metrics.Metrics()

options = options.get_options()

photon_id = 0

n_epochs = 1

# region: ............ Network ........... #

data_dir = 'data'
data_fn = 'SPY_1T_2016_2017'
data_res = 60

x_groups_on = False

dirs_on = True
diag_on = False

msgs_on = {
    'init': True,
    'run_log': True,
    'photon_log': True}

# ------ Cols ------ #

f_cols = [
    ['','0'],
    ['F1_hold_mins','1T'],
    ['F2_hold_mins','2T'],
    ['F3_hold_mins','3T'],
    ['F4_hold_mins','4T'],
    ['F5_hold_mins','5T'],
    ['F6_hold_mins','10T'],
    ['F7_hold_mins','15T'],
    ['F8_hold_mins','30T'],
    ['F9_hold_mins','1H'],
    ['F10_hold_mins','2H'],
    ['F11_hold_mins','3H'],
    ['F12_hold_mins','1D'],
    ['F13_hold_mins','3D'],
    ['F14_hold_mins','5D'],
    ['F15_hold_mins','7D']]

x_pcts = {
    'X_TP_VWAP': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_ROC_15T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_ROC_60T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_ROC_195T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_ROC_1D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_ROC_5D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_ROC_15D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_ROC_30D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_SMA_ROC_15T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_SMA_ROC_60T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_SMA_ROC_195T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_SMA_ROC_1D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_SMA_ROC_5D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_SMA_ROC_15D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_SMA_ROC_30D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True}
}

x_pcts_2 = {
    'X_TP_VWAP': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_ROC_195T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_ROC_1D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_ROC_5D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_SMA_ROC_195T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_SMA_ROC_1D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'X_SMA_ROC_5D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True}
}

x_cols_2 = {
    'bar_tp': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'bar_vwap': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'SMA_15T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'SMA_60T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'SMA_195T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'SMA_1D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'SMA_5D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'SMA_15D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'SMA_30D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'LAG_15T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'LAG_60T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'LAG_195T': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'LAG_1D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'LAG_5D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'LAG_15D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},

    'LAG_30D': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True}
}

ar_cols = {
    'bar_tp': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'bar_vwap': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'tf_yq': {
        'seq_agg': 'mean',
        'ofs_on': False,
        'nor_on': False},
    'tf_ym': {
        'seq_agg': 'mean',
        'ofs_on': False,
        'nor_on': False},
    'tf_yb': {
        'seq_agg': 'mean',
        'ofs_on': False,
        'nor_on': False},
    'tf_mb': {
        'seq_agg': 'mean',
        'ofs_on': False,
        'nor_on': False},
    'tf_wd': {
        'seq_agg': 'mean',
        'ofs_on': False,
        'nor_on': False},
    'tf_dh': {
        'seq_agg': 'mean',
        'ofs_on': False,
        'nor_on': False},
    'tf_dh': {
        'seq_agg': 'mean',
        'ofs_on': False,
        'nor_on': False},
    'tf_dp': {
        'seq_agg': 'mean',
        'ofs_on': False,
        'nor_on': False},
    'org_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'bar_date': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'st_time': {
        'seq_agg': 'first',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'ed_time': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'day_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'intra_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'block_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    }
}

y_cols_full = {
    'DB1_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB2_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB3_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB4_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB5_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB6_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB7_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},

    'DB1_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB2_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB3_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB4_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB5_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB6_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB7_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},

    'DB1_bar_date': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB2_bar_date': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB3_bar_date': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB4_bar_date': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB5_bar_date': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB6_bar_date': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB7_bar_date': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},

    'DB1_bar_time': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB2_bar_time': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB3_bar_time': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB4_bar_time': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB5_bar_time': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB6_bar_time': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB7_bar_time': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False}}

# --- x cols ---- #

pr_cols = {
    'BAR_TP': {'seq_agg': 'mean',
               'ofs_on': True,
               'nor_on': True,
               'x_group': 'pr'},

    'BAR_VWAP': {'seq_agg': 'mean',
                 'ofs_on': True,
                 'nor_on': True,
                 'x_group': 'pr'},

    'LAG_1D': {'seq_agg': 'mean',
               'ofs_on': True,
               'nor_on': True,
               'x_group': 'pr'},

    'LAG_5D': {'seq_agg': 'mean',
               'ofs_on': True,
               'nor_on': True,
               'x_group': 'pr'},

    'LAG_15D': {'seq_agg': 'mean',
                'ofs_on': True,
                'nor_on': True,
                'x_group': 'pr'},

    'LAG_30D': {'seq_agg': 'mean',
                'ofs_on': True,
                'nor_on': True,
                'x_group': 'pr'},

    'SMA_1D': {'seq_agg': 'mean',
               'ofs_on': True,
               'nor_on': True,
               'x_group': 'pr'},

    'SMA_5D': {'seq_agg': 'mean',
               'ofs_on': True,
               'nor_on': True,
               'x_group': 'pr'},

    'SMA_15D': {'seq_agg': 'mean',
                'ofs_on': True,
                'nor_on': True,
                'x_group': 'pr'},

    'SMA_30D': {'seq_agg': 'mean',
                'ofs_on': True,
                'nor_on': True,
                'x_group': 'pr'}}

vol_cols = {
    'BAR_VOL': {'seq_agg': 'mean',
                'ofs_on': True,
                'nor_on': True,
                'x_group': 'vol'},

    'VOL_1D': {'seq_agg': 'mean',
               'ofs_on': True,
               'nor_on': True,
               'x_group': 'vol'},

    'VOL_5D': {'seq_agg': 'mean',
               'ofs_on': True,
               'nor_on': True,
               'x_group': 'vol'},

    'VOL_15D': {'seq_agg': 'mean',
                'ofs_on': True,
                'nor_on': True,
                'x_group': 'vol'},

    'VOL_30D': {'seq_agg': 'mean',
                'ofs_on': True,
                'nor_on': True,
                'x_group': 'vol'}}

atr_cols = {
    'ATR_1D': {'seq_agg': 'mean',
               'ofs_on': True,
               'nor_on': True,
               'x_group': 'atr'},

    'ATR_5D': {'seq_agg': 'mean',
               'ofs_on': True,
               'nor_on': True,
               'x_group': 'atr'},

    'ATR_15D': {'seq_agg': 'mean',
                'ofs_on': True,
                'nor_on': True,
                'x_group': 'atr'},

    'ATR_30D': {'seq_agg': 'mean',
                'ofs_on': True,
                'nor_on': True,
                'x_group': 'atr'}}

roc_cols = {
    'ROC_VWAP': {'seq_agg': 'mean',
                 'ofs_on': True,
                 'nor_on': True,
                 'x_group': 'roc'},

    'ROC_1D': {'seq_agg': 'mean',
               'ofs_on': True,
               'nor_on': True,
               'x_group': 'roc'},

    'ROC_5D': {'seq_agg': 'mean',
               'ofs_on': True,
               'nor_on': True,
               'x_group': 'roc'},

    'ROC_15D': {'seq_agg': 'mean',
                'ofs_on': True,
                'nor_on': True,
                'x_group': 'roc'},

    'ROC_30D': {'seq_agg': 'mean',
                'ofs_on': True,
                'nor_on': True,
                'x_group': 'roc'},

    'ROC_SMA_1D': {'seq_agg': 'mean',
                   'ofs_on': True,
                   'nor_on': True,
                   'x_group': 'roc'},

    'ROC_SMA_5D': {'seq_agg': 'mean',
                   'ofs_on': True,
                   'nor_on': True,
                   'x_group': 'roc'},

    'ROC_SMA_15D': {'seq_agg': 'mean',
                    'ofs_on': True,
                    'nor_on': True,
                    'x_group': 'roc'},

    'ROC_SMA_30D': {'seq_agg': 'mean',
                    'ofs_on': True,
                    'nor_on': True,
                    'x_group': 'roc'}}

zsc_cols = {
    'ZSC_SMA_1D': {'seq_agg': 'mean',
                   'ofs_on': True,
                   'nor_on': True,
                   'x_group': 'zsc'},

    'ZSC_SMA_5D': {'seq_agg': 'mean',
                   'ofs_on': True,
                   'nor_on': True,
                   'x_group': 'zsc'},

    'ZSC_SMA_15D': {'seq_agg': 'mean',
                    'ofs_on': True,
                    'nor_on': True,
                    'x_group': 'zsc'},

    'ZSC_SMA_30D': {'seq_agg': 'mean',
                    'ofs_on': True,
                    'nor_on': True,
                    'x_group': 'zsc'}}

x_cols = pr_cols
# {**pr_cols, **vol_cols, **atr_cols, **roc_cols, **zsc_cols}

c_cols = {
    'tf_yq': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'tf_ym': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'tf_yb': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'tf_mb': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'tf_wd': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'tf_dh': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True},
    'tf_dp': {
        'seq_agg': 'mean',
        'ofs_on': True,
        'nor_on': True}
}

y_pcts = {
    'DB1_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB2_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB3_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB4_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB5_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB6_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB7_pct': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},

    'DB1_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB2_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB3_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB4_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB5_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB6_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB7_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False}}

db1_cls = {
    'DB1_S2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB1_S1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB1_N0': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB1_L1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB1_L2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False}}

db2_cls = {
    'DB2_S2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB2_S1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB2_N0': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB2_L1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB2_L2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False}}

db3_cls = {
    'DB3_S2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB3_S1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB3_N0': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB3_L1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB3_L2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False}}

db4_cls = {
    'DB4_S2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB4_S1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB4_N0': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB4_L1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB4_L2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False}}

db5_cls = {
    'DB5_S2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB5_S1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB5_N0': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB5_L1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB5_L2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False}}

db6_cls = {
    'DB6_S2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB6_S1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB6_N0': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB6_L1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB6_L2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False}}

db7_cls = {
    'DB7_S2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB7_S1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB7_N0': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB7_L1': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB7_L2': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False}}

y_tracking = {
    'DB1_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB2_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB3_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB4_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB5_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB6_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False},
    'DB7_bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'nor_on': False}}

y_cols = {**db2_cls, **y_tracking}

t_cols = {
    'bar_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'bar_ts': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'bar_date': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'bar_day': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'bar_st': {
        'seq_agg': 'first',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'bar_ed': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'day_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'intra_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    },
    'block_idx': {
        'seq_agg': 'last',
        'ofs_on': False,
        'sca_on': False,
        'nor_on': False
    }
}

data_cols = {
    'x_cols': x_cols,
    'c_cols': c_cols,
    'y_cols': y_cols,
    't_cols': t_cols,
    'f_cols': ['x_cols', 'c_cols']}

float_x = 32

network_config = {'photon_id': photon_id,
                  'data_dir': data_dir,
                  'data_fn': data_fn,
                  'data_res': data_res,
                  'data_cols':data_cols,
                  'x_groups_on':x_groups_on,
                  'dirs_on': dirs_on,
                  'diag_on': diag_on,
                  'msgs_on': msgs_on,
                  'float_x': float_x}

# endregion:

# region: ............ Tree ............. #

batch_size = 100

train_days = 100
test_days = 50
val_days = 100

outputs_on = True

samples_pd = 2
seed = None

shuffle = {'shuffle_on': False,
           'seed': seed}

masking = {'blocks': {'train': {'mask': utils.config_block_mask(),
                                'config': {'only_samples': True,
                                           'mask_tracking': False,
                                           'pre_apply': False,
                                           'pre_loss_apply': False}},

                      'test': {'mask': utils.config_block_mask(),
                               'config': {'only_samples': True,
                                          'mask_tracking': True,
                                          'pre_apply': False,
                                          'pre_loss_apply': False}},

                      'val': {'mask': utils.config_block_mask(),
                              'config': {'only_samples': True,
                                         'mask_tracking': True,
                                         'pre_apply': False,
                                         'pre_loss_apply': False}}}}

seq_days = 1

len = 390
agg = 5

val_on = True
test_on = False

offset = {'type': None,
          'x_on': False,
          'c_on': False,
          'y_on': False,
          't_on': False,
          'periods': 1}

normalize = {'x_cols': options.input_norm_args.min_max_scaler,
             'c_cols': options.input_norm_args.min_max_scaler,
             'y_cols': None,
             't_cols': None}

preproc = {'pre_agg': False,
           'offset': offset,
           'normalize': normalize}

tree_config = {'name': 'Base',
               'batch_size': batch_size,
               'shuffle': shuffle,
               'preproc': preproc,
               'samples_pd': samples_pd,
               'train_days': train_days,
               'test_days': test_days,
               'val_days': val_days,
               'seq_days': seq_days,
               'seq_len': len,
               'seq_agg': agg,
               'val_on': val_on,
               'test_on': test_on,
               'masking': masking,
               'outputs_on': outputs_on,
               'seed': seed}

# endregion:

# region: ------------------------ Branch Configs ------------------------  #

log_config = {'data': {'log_batch_data': {'main': False, 'val': False}},
              'models': {'log_calls': {'main': False, 'val': False},
                         'log_layers': {'main': True, 'val': False},
                         'log_run_data': {'main': False, 'val': False},
                         'log_theta': True}}

data_config = [{'input_src': 'batch',
                'targets': {'is_seq': False,
                            'split_on': 5},
                'log_config': log_config['data']}]

build_config = [{'strat_type': None,
                 'dist_type': None,
                 'pre_build': True,
                 'load_cp': True,
                 'save_cp': True}]

opt_config = [{'fn': optimizers.AdamDynamic,
               'args': {'lr_st': 0.01,
                        'lr_min': 1e-7,
                        'decay_rate': 1.25,
                        'static_epochs': [2,2]}}]

loss_config = [{'fn': losses.categorical_crossentropy,
                'args': {'from_logits': False,
                         'reduction': 'NONE'}}]

metrics_config = [{'fn': metrics.AUC,
                   'args': {"from_logits": False}},
                  {'fn': metrics.Precision,
                   'args': {}},
                  {'fn': metrics.Recall,
                   'args': {}}]

run_config = [{'run_type': 'fit',
               'data_type': 'train',
               'val_on': True,
               'metrics_on': True,
               'pre_build': True,
               'load_cp': True,
               'save_cp': True,
               'async_on': False,
               'msgs_on': True}]

save_config = [{'features': None,
                'x_tracking': None,
                'y_true': 'last',
                'y_hat': 'last',
                'y_tracking': None,
                'step_loss': 'All',
                'model_loss': 'All',
                'full_loss': 'All',
                'metrics': 'All',
                'preds': None,
                'grads': None,
                'learning_rates': 'All'}]

# endregion:

# region: ........ Transformers ........ #

trans_n_chains = 1

trans_model_config = [{'model': trans_models.Transformer_2,
                       'n_models': 1,
                       'n_outputs': 5,
                       'args': {'d_model': 512,
                                'reg_args': None,
                                'norm_args': None,
                                'reg_vals': [0],
                                'seed': seed,
                                'is_prob': False,
                                'show_calls': False,
                                'log_config': log_config['models']}}]

trans_config = {'name': 'trans',
                'n_epochs': n_epochs,
                'n_chains': trans_n_chains,
                'model_config': trans_model_config,
                'data_config': data_config,
                'build_config': build_config,
                'opt_config': opt_config,
                'loss_config': loss_config,
                'metrics_config': metrics_config,
                'run_config': run_config,
                'save_config': save_config}

# endregion:

# region: ........ CNN Models ........ #

cnn_n_chains = 1

cnn_model_config = [{'model': cnn_models.CNN_Base,
                     'n_models': 1,
                     'n_outputs': 5,
                     'args': {'d_model': 512,
                              'reg_args': None,
                              'norm_args': None,
                              'reg_vals': [0],
                              'seed': seed,
                              'is_prob': False,
                              'show_calls': False,
                              'log_config': log_config['models']}}]

cnn_config = {'name': 'CNN',
              'n_epochs': n_epochs,
              'n_chains': cnn_n_chains,
              'model_config': cnn_model_config,
              'data_config': data_config,
              'build_config': build_config,
              'opt_config': opt_config,
              'loss_config': loss_config,
              'metrics_config': metrics_config,
              'run_config': run_config,
              'save_config': save_config}

# endregion:

# region: ........ RNN Models ........ #

rnn_n_chains = 1

rnn_model_config = [{'model': rnn_models.LSTM_Pool,
                     'n_models': 1,
                     'n_outputs': 5,
                     'args': {'d_model': 512,
                              'reg_args': None,
                              'norm_args': None,
                              'reg_vals': [0],
                              'seed': seed,
                              'is_prob': False,
                              'show_calls': False,
                              'log_config': log_config['models']}}]

rnn_config = {'name': 'RNN',
              'n_epochs': n_epochs,
              'n_chains': rnn_n_chains,
              'model_config': rnn_model_config,
              'data_config': data_config,
              'build_config': build_config,
              'opt_config': opt_config,
              'loss_config': loss_config,
              'metrics_config': metrics_config,
              'run_config': run_config,
              'save_config': save_config}

# endregion:

# region: .......... Ensemble Learning .......... #

ens_n_chains = 5

ens_model_config = [{'model': ens_models.Model_A,
                     'n_models': 10,
                     'n_outputs': 5,
                     'args': {'d_model': 32,
                              'reg_args': options.model_reg_args.gauss_noise,
                              'norm_args': options.model_norm_args.batch,
                              'reg_vals': [.75,],
                              'seed': seed,
                              'is_prob': False,
                              'show_calls': False,
                              'log_config': log_config['models']}},

                    {'model': ens_models.Model_B,
                     'n_models': 10,
                     'n_outputs': 5,
                     'args': {'d_model': 32,
                              'reg_args': options.model_reg_args.gauss_noise,
                              'norm_args': options.model_norm_args.batch,
                              'reg_vals': [.75,],
                              'seed': seed,
                              'is_prob': False,
                              'show_calls': False,
                              'log_config': log_config['models']}},

                    {'model': ens_models.Model_B,
                     'n_models': 10,
                     'n_outputs': 5,
                     'args': {'d_model': 32,
                              'reg_args': options.model_reg_args.gauss_noise,
                              'norm_args': options.model_norm_args.batch,
                              'reg_vals': [.75,],
                              'seed': seed,
                              'is_prob': False,
                              'show_calls': False,
                              'log_config': log_config['models']}},

                    {'model': ens_models.Model_B,
                     'n_models': 10,
                     'n_outputs': 5,
                     'args': {'d_model': 32,
                              'reg_args': options.model_reg_args.gauss_noise,
                              'norm_args': options.model_norm_args.batch,
                              'reg_vals': [.75,],
                              'seed': seed,
                              'is_prob': False,
                              'show_calls': False,
                              'log_config': log_config['models']}},

                    {'model': ens_models.Model_C,
                     'n_models': 1,
                     'n_outputs': 5,
                     'args': {'d_model': 5,
                              'reg_args': options.model_reg_args.gauss_noise,
                              'norm_args': options.model_norm_args.batch,
                              'reg_vals': [.75,],
                              'seed': seed,
                              'is_prob': False,
                              'show_calls': False,
                              'log_config': log_config['models']}}]

ens_opt_config = [{'fn': optimizers.AdamDynamic,
                   'args': {'lr_st': 0.025,
                            'lr_min': 1e-7,
                            'decay_rate': 1.25,
                            'static_epochs': [1,2]}},

                  {'fn': optimizers.AdamDynamic,
                   'args': {'lr_st': 0.02,
                            'lr_min': 1e-7,
                            'decay_rate': 1.25,
                            'static_epochs': [1,2]}},

                  {'fn': optimizers.AdamDynamic,
                   'args': {'lr_st': 0.015,
                            'lr_min': 1e-7,
                            'decay_rate': 1.25,
                            'static_epochs': [1, 2]}},

                  {'fn': optimizers.AdamDynamic,
                   'args': {'lr_st': 0.015,
                            'lr_min': 1e-7,
                            'decay_rate': 1.25,
                            'static_epochs': [1, 2]}},

                  {'fn': optimizers.AdamDynamic,
                   'args': {'lr_st': 0.01,
                            'lr_min': 1e-7,
                            'decay_rate': 1.25,
                            'static_epochs': [1,2]}}]

ens_build_config = [{'strat_type': None,
                     'dist_type': None,
                     'pre_build': True,
                     'load_cp': True,
                     'save_cp': True}]

ens_loss_config = [{'fn': losses.categorical_crossentropy,
                    'args': {'from_logits': True,
                             'reduction': 'NONE'}}]

ens_metrics_config = [{'fn': photon_metrics.CatAcc,
                       'args': {}}]

ens_save_config = [{'features': None,
                    'x_tracking': None,
                    'y_true': 'last',
                    'y_hat': 'last',
                    'y_tracking': None,
                    'step_loss': 'All',
                    'model_loss': 'All',
                    'full_loss': 'All',
                    'metrics': 'All',
                    'preds': None,
                    'grads': None,
                    'learning_rates': 'All'}]

ens_run_config = [{'run_type': 'fit',
                   'data_type': 'train',
                   'val_on': True,
                   'metrics_on': True,
                   'pre_build': True,
                   'load_cp': True,
                   'save_cp': True,
                   'async_on': False,
                   'msgs_on': True}]

ens_config = {'name': 'Ens',
              'n_epochs': n_epochs,
              'n_chains': ens_n_chains,
              'model_config': ens_model_config,
              'data_config': data_config,
              'build_config': ens_build_config,
              'opt_config': ens_opt_config,
              'loss_config': ens_loss_config,
              'metrics_config':ens_metrics_config,
              'run_config': ens_run_config,
              'save_config': ens_save_config}

# endregion:

# region: ........ Probabilistic Models ........ #

prob_n_chains = 1

prob_model_config = [{'model': prob_models.Prob_1,
                       'n_models': 1,
                       'n_outputs': 5,
                       'args': {'d_model': 512,
                                'reg_args': None,
                                'norm_args': None,
                                'reg_vals': [0],
                                'seed': seed,
                                'is_prob': False,
                                'show_calls': False,
                                'log_config': log_config['models']}}]

prob_config = {'name': 'prob',
                'n_epochs': n_epochs,
                'n_chains': prob_n_chains,
                'model_config': prob_model_config,
                'data_config': data_config,
                'build_config': build_config,
                'opt_config': opt_config,
                'loss_config': loss_config,
                'metrics_config': metrics_config,
                'run_config': run_config,
                'save_config': save_config}

# endregion:
