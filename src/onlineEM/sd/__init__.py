from .gmm import gmm, gmm_params, gmm_stats
from .stm import stm, stm_params, stm_stats

models_zoo = {
    "GMM": gmm,
    "STM": stm,
}

params_zoo = {
    "GMM": gmm_params,
    "STM": stm_params,
}

stats_zoo = {
    "GMM": gmm_stats,
    "STM": stm_stats,
}

__all__ = [
    "gmm",
    "models_zoo",
    "gmm_params",
    "gmm_stats",
    "stm",
    "stm_params",
    "stm_stats",
]
