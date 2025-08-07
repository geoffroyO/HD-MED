import os
from onlineEM.core import Config, em_config
from onlineEM.data import create_dataloader, NumpyDataSource
from onlineEM.models import HDlm


def get_config(n_components: int = None, name: str = None):
    if name is None:
        name = f"hdlm_{n_components}"

    cfg = Config()
    batch_size = 1024

    cfg.em_config = em_config(
        n_components=n_components,
        num_features=260,
        num_epochs=10,
        batch_size=batch_size,
        n_first=20000,
    )

    cfg.schedule = lambda k: (1 - 10e-10) * (k + 1) ** (-6 / 10)

    cfg.model = HDlm()

    cfg.data_path = os.path.expandvars("$WORK/data/AOAS/DICO_noisy.npz")
    cfg.data_clean_path = os.path.expandvars("$WORK/data/AOAS/DICO.npz")

    cfg.data = create_dataloader(
        data_source=NumpyDataSource,
        file_path=cfg.data_path,
        test_split=0.2,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        num_epochs=1,
    )

    cfg.validation_interval = 200
    cfg.checkpoint_dir = os.path.expandvars(f"$SCRATCH/onlineEM/{name}/checkpoints")
    cfg.log_interval = 5
    cfg.polyak_update_frequency = 5
    return cfg
