from pathlib import Path
import orbax.checkpoint as ocp
from onlineEM.core import em_params, em_stats, em_config
import shutil
from typing import Tuple, Optional


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_to_keep = max_to_keep
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
            enable_async_checkpointing=True,
        )
        self.manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            options=options,
        )

    def save_config(self, config_file_path: str) -> None:
        """Save training config once at the beginning."""
        config_source = config_file_path

        # Copy the config file to checkpoint directory
        config_backup_path = self.checkpoint_dir / "config.py"
        shutil.copy2(config_source, config_backup_path)

    def save(
        self,
        step: int,
        params: em_params,
        stats: em_stats,
        config: em_config,
        polyak_params: Optional[em_params] = None,
    ):
        save_args = {
            "params": ocp.args.StandardSave(params),
            "stats": ocp.args.StandardSave(stats),
            "config": ocp.args.StandardSave(config),
        }

        save_args["polyak_params"] = ocp.args.StandardSave(polyak_params)

        self.manager.save(step, args=ocp.args.Composite(**save_args))

    def restore(
        self, blank_params_class: em_params, blank_stats_class: em_stats
    ) -> Tuple[em_params, em_stats, em_config, em_params]:
        step = self.manager.latest_step()
        restored = self.manager.restore(step)
        params = blank_params_class(**restored["params"])
        polyak_params = blank_params_class(**restored["polyak_params"])
        stats = blank_stats_class(**restored["stats"])
        config = em_config(**restored["config"])

        return params, stats, config, polyak_params

    def wait_until_finished(self):
        self.manager.wait_until_finished()
