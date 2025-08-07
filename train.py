import importlib.util
import os
import sys

import jax
from absl import app, flags

from onlineEM import Trainer

# Cache compilation
jax.config.update("jax_compilation_cache_dir", os.path.expandvars("$SCRATCH/online_em_cache"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

# GPU performance tips
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true"

# Flags
FLAGS = flags.FLAGS
flags.DEFINE_string("config", "examples/hdgmm.py", "Path to config file")
flags.DEFINE_integer("n_components", 30, "Number of components")


def main(_):
    spec = importlib.util.spec_from_file_location("config", FLAGS.config)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)
    config = config_module.get_config(FLAGS.n_components)
    trainer = Trainer(config, FLAGS.config)

    trainer_state = trainer.burnin()
    trainer_state = trainer.train(trainer_state)


if __name__ == "__main__":
    app.run(main)
