from .hdgmm import hdgmm, hdgmm_params, hdgmm_stats
from .hdstm import hdstm, hdstm_params, hdstm_stats

models_zoo = {
    "HDGMM": hdgmm,
    "HDSTM": hdstm,
}

params_zoo = {
    "HDGMM": hdgmm_params,
    "HDSTM": hdstm_params,
}

stats_zoo = {
    "HDGMM": hdgmm_stats,
    "HDSTM": hdstm_stats,
}

__all__ = [
    "hdgmm",
    "hdstm",
    "models_zoo",
    "hdgmm_params",
    "hdgmm_stats",
    "hdstm_params",
    "hdstm_stats",
]
