from onlineEM.core import Config, em_params, em_stats
import numpy as np
import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from tqdm import tqdm
import time
from typing import Tuple, Callable, NamedTuple
from pathlib import Path
from .. import logger
from ..utils import create_em_logger, CheckpointManager, size_params
from ..utils.polyak import PolyakAverager, PolyakAveragerState


class TrainerState(NamedTuple):
    """Extended training state that includes Polyak averaged parameters."""

    em_params: em_params
    em_stats: em_stats
    polyak_state: PolyakAveragerState


class Trainer:
    def __init__(self, config: Config, config_file_path: str):
        self.config = config

        self.schedule = self.config.schedule
        self.model = self.config.model

        self.data_loader_train = self.config.data.train_loader
        self.data_loader_test = self.config.data.test_loader
        self.batch_size = self.config.data.batch_size
        self.num_train_steps = self.config.data.num_train_steps
        self.num_val_steps = self.config.data.num_val_steps
        self.num_epochs = self.config.data.num_epochs

        self.validation_interval = self.config.validation_interval
        self.log_interval = self.config.log_interval

        self.polyak_averager = PolyakAverager(self.config.polyak_update_frequency)
        logger.info(f"📊 Polyak averaging with update frequency: {self.config.polyak_update_frequency}")

        self.burnin_step = self.create_burnin_step()
        self.train_step = self.create_train_step()
        self.val_step = self.create_val_step()

        checkpoint_dir = config.checkpoint_dir
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.checkpoint_manager.save_config(config_file_path)
        logger.info(f"📊 Checkpointer initialized: {checkpoint_dir}")

        hdf5_log_path = Path(checkpoint_dir) / "logger.hdf5"
        self.hdf5_logger = create_em_logger(hdf5_log_path)
        logger.info(f"📊 HDF5 logger initialized: {hdf5_log_path}")

        logger.info(
            f"🚀 Training {self.model.__class__.__name__} with n_components: {self.config.em_config.n_components}, num_features: {self.config.em_config.num_features}, num_epochs: {self.config.em_config.num_epochs}, batch_size: {self.config.em_config.batch_size} 🎯"
        )

    def init(self):
        logger.info("🔄 Initializing model... 🤖")
        start_time = time.time()
        n_first = self.config.em_config.n_first
        tmp_it = iter(self.data_loader_train)
        X = np.concatenate([next(tmp_it) for _ in range(n_first // self.batch_size)])
        self.em_config, em_params, em_stats = self.model.init(X, self.config.em_config)

        logger.info(
            f"✅ Model initialized successfully with reduction: {self.em_config.reduction} in {time.time() - start_time:.2f} seconds 🎉"
        )
        del tmp_it
        del X

        return em_params, em_stats

    def create_burnin_step(
        self,
    ):
        @jax.jit
        def burnin_step(batch: ArrayLike, step: int, params: em_params, stats: em_stats) -> em_stats:
            stats = self.model.burnin(batch, step, params, stats, self.em_config, self.schedule)
            return stats

        return burnin_step

    def create_train_step(
        self,
    ):
        @jax.jit
        def train_step(batch: ArrayLike, step: int, params: em_params, stats: em_stats) -> Tuple[em_params, em_stats]:
            params, stats = self.model.update(batch, step, params, stats, self.em_config, self.schedule)
            return params, stats

        return train_step

    def create_val_step(self) -> Callable[[ArrayLike, em_params], Array]:
        @jax.jit
        def val_step(batch: ArrayLike, params: em_params) -> Array:
            log_prob = self.model.batch_log_prob(batch, params, self.em_config)
            return log_prob

        return val_step

    def burnin(self) -> TrainerState:
        em_params, em_stats = self.init()
        logger.info("🔥 Burning in model... 🔥")
        start_time = time.time()
        burning_iter = iter(self.data_loader_train)
        for step in tqdm(range(2 * self.em_config.num_features), desc="🔥 Burnin Progress", unit="step"):
            batch = next(burning_iter)
            em_stats = self.burnin_step(batch, step, em_params, em_stats)
        logger.info(f"🎊 Burnin completed successfully in {time.time() - start_time:.2f} seconds! 🎊")

        # Initialize Polyak state after burnin
        polyak_state = self.polyak_averager.init_state(em_params)
        logger.info("📊 Polyak averaging state initialized")

        return TrainerState(em_params, em_stats, polyak_state)

    def validate(self, em_params: em_params) -> float:
        """Run validation and return average validation loss.

        Args:
            em_params: Current EM parameters

        Returns:
            Average validation loss
        """
        logger.info("🔄 Validating model... 🤖")
        val_loss = 0.0
        val_iter = iter(self.data_loader_test)
        bic_val, bic_count = 0, 0
        for _ in range(self.num_val_steps):
            batch = next(val_iter)
            log_prob = self.val_step(batch, em_params)
            val_loss += log_prob.mean()
            bic_val += log_prob.sum()
            bic_count += len(batch)

        val_loss /= self.num_val_steps
        logger.info(f"📈 Validation Loss: {val_loss:.2f}")

        num_params = sum(size_params(array) for array in em_params)
        bic = -2 * bic_val + num_params * jnp.log(bic_count)
        logger.info(f"📈 BIC: {bic:.2f}")
        return float(val_loss)

    def train(self, trainer_state: TrainerState) -> TrainerState:
        logger.info("🔥 Training model... 🔥")
        start_time = time.time()
        train_iter = iter(self.data_loader_train)

        em_params = trainer_state.em_params
        em_stats = trainer_state.em_stats
        polyak_state = trainer_state.polyak_state

        # Calculate steps per epoch for logging
        steps_per_epoch = self.num_train_steps // self.num_epochs if self.num_epochs > 0 else self.num_train_steps

        for step in tqdm(range(self.num_train_steps), desc="🔥 Training Progress", unit="step"):
            batch = next(train_iter)
            em_params, em_stats = self.train_step(batch, step, em_params, em_stats)

            polyak_state = self.polyak_averager.update_if_needed(step, polyak_state, em_params)

            # Calculate current epoch
            current_epoch = step // steps_per_epoch if steps_per_epoch > 0 else 0

            # Validation and logging
            val_loss = None
            if (step + 1) % self.validation_interval == 0:
                val_loss = self.validate(em_params)
                # Save with Polyak state if available
                self.checkpoint_manager.save(step, em_params, em_stats, self.em_config, polyak_state.params_polyak)

            # Log only at specified intervals
            if (step + 1) % self.log_interval == 0:
                # Log both regular and Polyak parameters
                log_params = em_params
                log_polyak_params = polyak_state.params_polyak

                self.hdf5_logger.log_step(
                    step=step,
                    epoch=current_epoch,
                    params=log_params,
                    stats=em_stats,
                    val_loss=val_loss,
                    polyak_params=log_polyak_params,
                )

        self.hdf5_logger.shutdown()
        logger.info("📊 HDF5 logger closed")

        # Final checkpoint save
        self.checkpoint_manager.save(step, em_params, em_stats, self.em_config, polyak_state.params_polyak)
        self.checkpoint_manager.wait_until_finished()
        logger.info("📊 Checkpointer closed")

        logger.info(f"🎊 Training completed successfully in {time.time() - start_time:.2f} seconds! 🎊")
        return TrainerState(em_params, em_stats, polyak_state)
