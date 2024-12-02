from .. import em_config
from ..hd import hdgmm_params, hdgmm_stats, hdstm_params, hdstm_stats
from ..sd import gmm_params, gmm_stats, stm_params, stm_stats

import numpy as np


def save_model(params, stat, config, save_path):
    params_dict = params._asdict()
    stat_dict = stat._asdict()
    config_dict = config._asdict()

    model = {
        "params": params_dict,
        "stat": stat_dict,
        "config": config_dict,
    }
    np.save(save_path, model)


def load_model(load_path, model_type):
    model = np.load(load_path, allow_pickle=True).item()

    params = model["params"]
    stat = model["stat"]
    config = model["config"]

    config = em_config(**config)

    if model_type == "hdgmm":
        params = hdgmm_params(**params)
        stat = hdgmm_stats(**stat)

    elif model_type == "hdstm":
        params = hdstm_params(**params)
        stat = hdstm_stats(**stat)

    elif model_type == "gmm":
        params = gmm_params(**params)
        stat = gmm_stats(**stat)

    elif model_type == "stm":
        params = stm_params(**params)
        stat = stm_stats(**stat)

    return config, stat, params
