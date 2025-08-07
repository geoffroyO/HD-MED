"""Async HDF5 logger for EM training parameters and statistics."""

import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

from ..core.base import em_params, em_stats
from .. import logger


class AsyncHDF5Logger:
    """Async HDF5 logger for EM training data with native array storage."""

    def __init__(self, file_path: Union[str, Path]):
        """Initialize the async HDF5 logger."""
        self.file_path = Path(file_path)
        self.metrics_queue = queue.Queue()
        self.hdf5_writer_thread = None
        self._setup_file()
        self.start()
        logger.info(f"📁 AsyncHDF5Logger initialized: {self.file_path}")

    def _setup_file(self) -> None:
        """Setup the HDF5 file structure."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create initial file structure if it doesn't exist
        if not self.file_path.exists():
            with h5py.File(self.file_path, "w") as f:
                metadata_group = f.create_group("metadata")
                metadata_group.attrs["created"] = datetime.now().isoformat()
                metadata_group.attrs["format_version"] = "1.0"

    def start(self) -> None:
        """Start the async HDF5 writer thread."""
        self.hdf5_writer_thread = threading.Thread(target=self._hdf5_writer_worker, daemon=True)
        self.hdf5_writer_thread.start()

    def log_step(
        self,
        step: int,
        epoch: int,
        params: em_params,
        stats: em_stats,
        val_loss: Optional[float] = None,
        polyak_params: Optional[em_params] = None,
    ) -> None:
        """Log training step data asynchronously with native array storage."""
        log_data = {
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "val_loss": val_loss,
            "params": params,
            "polyak_params": polyak_params,
            "stats": stats,
        }

        try:
            self.metrics_queue.put(log_data.copy(), block=False)
        except queue.Full:
            logger.warning("⚠️ HDF5 logger queue full, dropping log entry")

    def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shutdown the async HDF5 writer."""
        try:
            self.metrics_queue.join()
            self.metrics_queue.put(None)
            if self.hdf5_writer_thread is not None:
                self.hdf5_writer_thread.join(timeout=timeout)
            logger.info("✅ AsyncHDF5Logger shutdown complete")
        except Exception as e:
            logger.warning(f"❌ Error during HDF5 logger shutdown: {e}")

    def _hdf5_writer_worker(self) -> None:
        """Worker thread that writes metrics to HDF5."""
        while True:
            try:
                log_data = self.metrics_queue.get()
                if log_data is None:
                    break
                self._write_hdf5_sync(log_data)
                self.metrics_queue.task_done()
            except Exception as e:
                logger.warning(f"❌ HDF5 writer thread error: {e}")

    def _write_hdf5_sync(self, log_data: Dict[str, Any]) -> None:
        """Synchronously write log data to HDF5 file."""
        try:
            with h5py.File(self.file_path, "a") as f:
                step = log_data["step"]
                step_group_name = f"step_{step:06d}"

                # Create step group if it doesn't exist
                if step_group_name in f:
                    step_group = f[step_group_name]
                else:
                    step_group = f.create_group(step_group_name)

                # Store metadata
                self._write_metadata(step_group, log_data)

                # Store parameters
                self._write_parameters(step_group, log_data["params"], "params")

                # Store Polyak parameters
                self._write_parameters(step_group, log_data["polyak_params"], "polyak_params")

                # Store statistics
                self._write_statistics(step_group, log_data["stats"])

        except Exception as e:
            logger.warning(f"❌ Failed to write to HDF5: {e}")

    def _write_metadata(self, step_group: h5py.Group, log_data: Dict[str, Any]) -> None:
        """Write metadata for a training step."""
        step_group.attrs["step"] = log_data["step"]
        step_group.attrs["epoch"] = log_data["epoch"]
        step_group.attrs["timestamp"] = log_data["timestamp"]
        if log_data["val_loss"] is not None:
            step_group.attrs["val_loss"] = log_data["val_loss"]

    def _write_parameters(self, step_group: h5py.Group, params: em_params, group_name: str) -> None:
        """Write parameters to HDF5 group."""
        if group_name not in step_group:
            params_group = step_group.create_group(group_name)
        else:
            params_group = step_group[group_name]

        for field_name in params._fields:
            value = getattr(params, field_name)
            self._write_parameter_field(params_group, field_name, value)

    def _write_parameter_field(self, group: h5py.Group, field_name: str, value: Any) -> None:
        """Write a single parameter field to HDF5."""
        if isinstance(value, (jnp.ndarray, np.ndarray)):
            # Store arrays directly with proper shape
            array_data = np.asarray(value)
            if field_name in group:
                del group[field_name]
            # Only use compression for non-scalar arrays
            if array_data.ndim > 0:
                group.create_dataset(field_name, data=array_data, compression="gzip")
            else:
                group.create_dataset(field_name, data=array_data)

        elif isinstance(value, list):
            # Handle list of arrays (like A, D_tilde in hdgmm)
            self._write_parameter_list(group, field_name, value)
        else:
            # Store scalars as attributes
            group.attrs[field_name] = float(value)

    def _write_parameter_list(self, group: h5py.Group, field_name: str, value_list: List[Any]) -> None:
        """Write a list of parameters to HDF5."""
        if field_name in group:
            del group[field_name]
        list_group = group.create_group(field_name)

        for j, item in enumerate(value_list):
            if isinstance(item, (jnp.ndarray, np.ndarray)):
                array_data = np.asarray(item)
                # Only use compression for non-scalar arrays
                if array_data.ndim > 0:
                    list_group.create_dataset(f"item_{j}", data=array_data, compression="gzip")
                else:
                    list_group.create_dataset(f"item_{j}", data=array_data)
            else:
                list_group.attrs[f"item_{j}"] = item

    def _write_statistics(self, step_group: h5py.Group, stats: em_stats) -> None:
        """Write statistics to HDF5 group."""
        if "stats" not in step_group:
            stats_group = step_group.create_group("stats")
        else:
            stats_group = step_group["stats"]

        for field_name in stats._fields:
            value = getattr(stats, field_name)

            if isinstance(value, (jnp.ndarray, np.ndarray)):
                array_data = np.asarray(value)
                if field_name in stats_group:
                    del stats_group[field_name]
                # Only use compression for non-scalar arrays
                if array_data.ndim > 0:
                    stats_group.create_dataset(field_name, data=array_data, compression="gzip")
                else:
                    stats_group.create_dataset(field_name, data=array_data)
            else:
                stats_group.attrs[field_name] = float(value)


class HDF5LogReader:
    """Reader for HDF5 logs created by AsyncHDF5Logger."""

    def __init__(self, file_path: Union[str, Path]):
        """Initialize HDF5 log reader."""
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.file_path}")

    def get_steps(self) -> List[int]:
        """Get list of all logged steps."""
        with h5py.File(self.file_path, "r") as f:
            steps = []
            for key in f.keys():
                if key.startswith("step_"):
                    step_num = int(key.split("_")[1])
                    steps.append(step_num)
            return sorted(steps)

    def inspect_file_structure(self) -> None:
        """Debug function to inspect the HDF5 file structure."""
        with h5py.File(self.file_path, "r") as f:
            steps = self.get_steps()
            if steps:
                first_step = f[f"step_{steps[0]:06d}"]
                print(f"Groups in first step: {list(first_step.keys())}")

                if "params" in first_step:
                    print(f"Regular params: {list(first_step['params'].keys())}")

                if "polyak_params" in first_step:
                    print(f"Polyak params: {list(first_step['polyak_params'].keys())}")
                else:
                    print("No polyak_params found in HDF5 file")

                if "stats" in first_step:
                    print(f"Stats: {list(first_step['stats'].keys())}")
            else:
                print("No steps found in HDF5 file")

    def convergence_plot(
        self,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[Union[str, Path]] = None,
        plot_mode: str = "both",
        stats: bool = True,
    ) -> plt.Figure:
        """Plot parameters and statistics convergence.

        Args:
            figsize: Figure size tuple (width, height)
            save_path: Optional path to save the plot
            plot_mode: Controls which parameters to plot:
                - "both": Plot both regular and Polyak parameters (default)
                - "regular": Plot only regular parameters
                - "polyak": Plot only Polyak parameters
            stats: Whether to include statistics plots (default: True)

        Rules:
        - Size k: Plot k separate lines
        - Size k×n: Plot vector norms for k vectors
        - Size k×n×m: Plot matrix norms for k matrices
        """
        if plot_mode not in ["both", "regular", "polyak"]:
            raise ValueError(f"plot_mode must be one of ['both', 'regular', 'polyak'], got '{plot_mode}'")

        steps = self.get_steps()
        if not steps:
            raise ValueError("No training steps found in HDF5 file")

        # Extract parameter and statistics metadata
        plot_data = self._extract_parameter_metadata()

        # Extract evolution data for plotting
        evolution_data = self._extract_evolution_data(plot_data, steps)

        if not evolution_data:
            raise ValueError("No parameter or statistics data found in HDF5 file")

        # Group parameters and prepare data for plotting
        param_groups, stats_data = self._group_parameters(evolution_data)

        # Create and configure the plot
        fig, axes = self._create_plot_layout(param_groups, stats_data, plot_mode, stats, figsize)

        # Plot parameters and statistics
        plot_idx = self._plot_parameters(axes, param_groups, plot_data, steps, plot_mode)
        plot_idx = self._plot_statistics(axes, stats_data, plot_data, steps, plot_idx, stats)

        # Finalize plot
        self._finalize_plot(fig, axes, plot_idx, plot_mode, stats)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _extract_parameter_metadata(self) -> Dict[str, Any]:
        """Extract parameter shapes and structure from the first step."""
        plot_data = {}
        steps = self.get_steps()

        with h5py.File(self.file_path, "r") as f:
            first_step = f[f"step_{steps[0]:06d}"]

            # Extract regular parameters
            if "params" in first_step:
                self._extract_group_metadata(first_step["params"], "params", plot_data)

            # Extract Polyak parameters
            if "polyak_params" in first_step:
                self._extract_group_metadata(first_step["polyak_params"], "polyak_params", plot_data)

            # Extract statistics
            if "stats" in first_step:
                self._extract_group_metadata(first_step["stats"], "stats", plot_data)

        return plot_data

    def _extract_group_metadata(self, group: h5py.Group, prefix: str, plot_data: Dict[str, Any]) -> None:
        """Extract metadata from a specific HDF5 group."""
        for param_name in group.keys():
            if isinstance(group[param_name], h5py.Dataset):
                shape = group[param_name].shape
                plot_data[f"{prefix}_{param_name}"] = shape
            elif isinstance(group[param_name], h5py.Group):
                # Handle list of arrays (A, D_tilde)
                list_group = group[param_name]
                n_clusters = len(list_group.keys())
                first_item_shape = list_group["item_0"].shape

                if len(first_item_shape) == 1:
                    plot_data[f"{prefix}_{param_name}"] = (n_clusters, "list_1d")
                elif len(first_item_shape) == 2:
                    plot_data[f"{prefix}_{param_name}"] = (n_clusters, "list_2d")
                else:
                    plot_data[f"{prefix}_{param_name}"] = (n_clusters, f"list_{len(first_item_shape)}d")

    def _extract_evolution_data(self, plot_data: Dict[str, Any], steps: List[int]) -> Dict[str, np.ndarray]:
        """Extract parameter evolution data across all steps."""
        evolution_data = {}
        initial_data = {}  # Store initial matrices for distance computation

        with h5py.File(self.file_path, "r") as f:
            for data_name, shape in plot_data.items():
                evolution = []

                # Handle proper splitting for polyak_params
                if data_name.startswith("polyak_params_"):
                    group_type = "polyak_params"
                    field_name = data_name[14:]  # Remove "polyak_params_" prefix
                else:
                    group_type, field_name = data_name.split("_", 1)

                # Store initial data for distance computation
                self._store_initial_data(f, steps[0], data_name, shape, group_type, field_name, initial_data)

                # Extract data for each step
                for step in steps:
                    step_group = f[f"step_{step:06d}"]
                    if group_type in step_group and field_name in step_group[group_type]:
                        step_data = self._extract_step_data(
                            step_group[group_type][field_name], shape, data_name, initial_data
                        )
                        evolution.append(step_data)

                if evolution:
                    evolution_data[data_name] = np.array(evolution)

        return evolution_data

    def _store_initial_data(
        self,
        f: h5py.File,
        first_step: int,
        data_name: str,
        shape: Any,
        group_type: str,
        field_name: str,
        initial_data: Dict[str, Any],
    ) -> None:
        """Store initial matrices for distance computation."""
        if len(shape) == 3 or (
            isinstance(shape, tuple) and len(shape) == 2 and isinstance(shape[1], str) and shape[1] == "list_2d"
        ):
            first_step_group = f[f"step_{first_step:06d}"]
            if group_type in first_step_group and field_name in first_step_group[group_type]:
                if isinstance(shape, tuple) and len(shape) == 2 and isinstance(shape[1], str):
                    # Handle list of 2D arrays
                    list_group = first_step_group[group_type][field_name]
                    n_clusters = shape[0]
                    initial_matrices = []
                    for i in range(n_clusters):
                        initial_matrices.append(np.array(list_group[f"item_{i}"]))
                    initial_data[data_name] = initial_matrices
                else:
                    # Handle regular kxnxm arrays
                    initial_data[data_name] = np.array(first_step_group[group_type][field_name])

    def _extract_step_data(
        self, step_data: Union[h5py.Dataset, h5py.Group], shape: Any, data_name: str, initial_data: Dict[str, Any]
    ) -> np.ndarray:
        """Extract data for a single step based on parameter type."""
        if isinstance(shape, tuple) and len(shape) == 2 and isinstance(shape[1], str):
            return self._extract_list_data(step_data, shape, data_name, initial_data)
        else:
            return self._extract_array_data(step_data, shape, data_name, initial_data)

    def _extract_list_data(
        self, list_group: h5py.Group, shape: Tuple[int, str], data_name: str, initial_data: Dict[str, Any]
    ) -> np.ndarray:
        """Extract data from list of arrays."""
        n_clusters, list_type = shape
        norms = []

        for i in range(n_clusters):
            item_data = np.array(list_group[f"item_{i}"])

            if list_type == "list_1d":
                norms.append(np.linalg.norm(item_data))
            elif list_type == "list_2d":
                if data_name in initial_data:
                    initial_matrix = initial_data[data_name][i]
                    diff = item_data - initial_matrix
                    norms.append(np.linalg.norm(diff, "fro"))
                else:
                    norms.append(np.linalg.norm(item_data, "fro"))
            else:
                norms.append(np.linalg.norm(item_data.flatten()))

        return np.array(norms)

    def _extract_array_data(
        self, data: h5py.Dataset, shape: Tuple[int, ...], data_name: str, initial_data: Dict[str, Any]
    ) -> np.ndarray:
        """Extract data from regular arrays."""
        data_array = np.array(data)

        if len(shape) == 0:
            # Scalar data - return as is
            return data_array
        elif len(shape) == 1:
            return data_array
        elif len(shape) == 2:
            return np.linalg.norm(data_array, axis=1)
        elif len(shape) == 3:
            if data_name in initial_data:
                initial_matrices = initial_data[data_name]
                distances = []
                for i in range(shape[0]):
                    diff = data_array[i] - initial_matrices[i]
                    distances.append(np.linalg.norm(diff, "fro"))
                return np.array(distances)
            else:
                return np.linalg.norm(data_array.reshape(shape[0], -1), axis=1)
        else:
            if shape[0] > 0:
                return np.linalg.norm(data_array.reshape(shape[0], -1), axis=1)
            else:
                return data_array.flatten()

    def _group_parameters(
        self, evolution_data: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
        """Group regular and Polyak parameters together."""
        param_groups = {}
        stats_data = {}

        for data_name, data_array in evolution_data.items():
            if data_name.startswith("params_"):
                param_name = data_name[7:]  # Remove "params_" prefix
                if param_name not in param_groups:
                    param_groups[param_name] = {}
                param_groups[param_name]["regular"] = (data_name, data_array)
            elif data_name.startswith("polyak_params_"):
                param_name = data_name[14:]  # Remove "polyak_params_" prefix
                if param_name not in param_groups:
                    param_groups[param_name] = {}
                param_groups[param_name]["polyak"] = (data_name, data_array)
            elif data_name.startswith("stats_"):
                stats_data[data_name] = data_array

        return param_groups, stats_data

    def _create_plot_layout(
        self,
        param_groups: Dict[str, Dict[str, Any]],
        stats_data: Dict[str, np.ndarray],
        plot_mode: str,
        stats: bool,
        figsize: Tuple[int, int],
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Create the plot layout based on the number of parameters and stats."""
        # Count plots
        n_param_plots = sum(
            1
            for param_name, param_data in param_groups.items()
            if (
                (plot_mode == "regular" and "regular" in param_data)
                or (plot_mode == "polyak" and "polyak" in param_data)
                or (plot_mode == "both" and ("regular" in param_data or "polyak" in param_data))
            )
        )

        n_stats_plots = len(stats_data) if stats else 0
        n_plots = n_param_plots + n_stats_plots

        if n_plots == 0:
            if plot_mode == "polyak":
                with h5py.File(self.file_path, "r") as f:
                    steps = self.get_steps()
                    first_step = f[f"step_{steps[0]:06d}"]
                    if "polyak_params" not in first_step:
                        raise ValueError(
                            f"No Polyak parameters found in HDF5 file. "
                            f"This log file was likely created before Polyak averaging was enabled. "
                            f"Available groups: {list(first_step.keys())}"
                        )
            raise ValueError(f"No data available for plot_mode='{plot_mode}'")

        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        return fig, axes

    def _plot_parameters(
        self,
        axes: List[plt.Axes],
        param_groups: Dict[str, Dict[str, Any]],
        plot_data: Dict[str, Any],
        steps: List[int],
        plot_mode: str,
    ) -> int:
        """Plot parameter evolution curves."""
        plot_idx = 0

        for param_name, param_data in param_groups.items():
            ax = axes[plot_idx]

            # Plot regular parameters if requested and available
            if plot_mode in ["both", "regular"] and "regular" in param_data:
                data_name, data_array = param_data["regular"]
                shape = plot_data[data_name]
                self._plot_parameter_curves(ax, steps, data_array, param_name, shape, curve_type="regular")

            # Plot Polyak parameters if requested and available
            if plot_mode in ["both", "polyak"] and "polyak" in param_data:
                polyak_data_name, polyak_data_array = param_data["polyak"]
                polyak_shape = plot_data[polyak_data_name]
                self._plot_parameter_curves(ax, steps, polyak_data_array, param_name, polyak_shape, curve_type="polyak")

            # Skip this parameter if neither type is available/requested
            if (
                (plot_mode == "regular" and "regular" not in param_data)
                or (plot_mode == "polyak" and "polyak" not in param_data)
                or (plot_mode == "both" and "regular" not in param_data and "polyak" not in param_data)
            ):
                continue

            # Set common plot properties
            self._configure_axis(ax, param_data, plot_data, plot_mode)
            plot_idx += 1

        return plot_idx

    def _plot_statistics(
        self,
        axes: List[plt.Axes],
        stats_data: Dict[str, np.ndarray],
        plot_data: Dict[str, Any],
        steps: List[int],
        plot_idx: int,
        stats: bool,
    ) -> int:
        """Plot statistics evolution curves."""
        if not stats:
            return plot_idx

        for stats_name, stats_array in stats_data.items():
            ax = axes[plot_idx]
            shape = plot_data[stats_name]
            field_name = stats_name[6:]  # Remove "stats_" prefix

            self._plot_statistics_curves(ax, steps, stats_array, field_name, shape)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

            # Add legend if not too many lines
            if isinstance(shape, tuple) and len(shape) == 2 and isinstance(shape[1], str):
                if shape[0] <= 8:
                    ax.legend(fontsize=8)
            elif len(shape) == 0:
                # Always show legend for scalar statistics
                ax.legend(fontsize=8)
            elif len(shape) >= 1 and shape[0] <= 8:
                ax.legend(fontsize=8)

            plot_idx += 1

        return plot_idx

    def _plot_parameter_curves(
        self,
        ax: plt.Axes,
        steps: List[int],
        data_array: np.ndarray,
        param_name: str,
        shape: Any,
        curve_type: str = "regular",
    ) -> None:
        """Plot parameter curves with appropriate styling."""
        # Define styling
        if curve_type == "polyak":
            linestyle, alpha, linewidth, prefix = "-", 1.0, 1.2, "Polyak "
        else:
            linestyle, alpha, linewidth, prefix = "-", 0.7, 0.5, ""

        # Get colors
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        # Plot based on dimensionality
        if isinstance(shape, tuple) and len(shape) == 2 and isinstance(shape[1], str):
            self._plot_list_parameters(
                ax, steps, data_array, param_name, shape, linestyle, alpha, linewidth, prefix, colors, curve_type
            )
        else:
            self._plot_array_parameters(
                ax, steps, data_array, param_name, shape, linestyle, alpha, linewidth, prefix, colors, curve_type
            )

    def _plot_list_parameters(
        self,
        ax: plt.Axes,
        steps: List[int],
        data_array: np.ndarray,
        param_name: str,
        shape: Tuple[int, str],
        linestyle: str,
        alpha: float,
        linewidth: float,
        prefix: str,
        colors: List[str],
        curve_type: str,
    ) -> None:
        """Plot list parameters (A, D_tilde)."""
        k, list_type = shape

        for i in range(k):
            color_idx = i % len(colors)
            color = colors[color_idx]  # Use same colors for both regular and Polyak

            if list_type == "list_1d":
                label = f"{prefix}||{param_name}[{i}]||"
            elif list_type == "list_2d":
                label = f"{prefix}||{param_name}[{i} - {param_name}₀[{i}]||_F"
            else:
                label = f"{prefix}||{param_name}[{i}]||"

            ax.plot(
                steps, data_array[:, i], label=label, linewidth=linewidth, linestyle=linestyle, alpha=alpha, color=color
            )

        # Set title
        if list_type == "list_1d":
            ax.set_title(f"||{param_name}||")
        elif list_type == "list_2d":
            ax.set_title(f"||{param_name} - {param_name}_0||_F")
        else:
            ax.set_title(f"||{param_name}||")

    def _plot_array_parameters(
        self,
        ax: plt.Axes,
        steps: List[int],
        data_array: np.ndarray,
        param_name: str,
        shape: Tuple[int, ...],
        linestyle: str,
        alpha: float,
        linewidth: float,
        prefix: str,
        colors: List[str],
        curve_type: str,
    ) -> None:
        """Plot regular array parameters."""
        if len(shape) == 0:
            # Scalar parameter
            color = colors[0]  # Use same color for both regular and Polyak
            label = f"{prefix}{param_name}"
            title = f"{param_name}"
            ax.plot(steps, data_array, label=label, linewidth=linewidth, linestyle=linestyle, alpha=alpha, color=color)
        else:
            for i in range(shape[0]):
                color_idx = i % len(colors)
                color = colors[color_idx]  # Use same colors for both regular and Polyak

                if len(shape) == 1:
                    label = f"{prefix}{param_name}[{i}]"
                    title = f"{param_name}"
                elif len(shape) == 2:
                    label = f"{prefix}||{param_name}[{i}]||"
                    title = f"||{param_name}||"
                elif len(shape) == 3:
                    label = f"{prefix}||{param_name}[{i} - {param_name}₀[{i}]||_F"
                    title = f"||{param_name} - {param_name}_0||_F"
                else:
                    label = f"{prefix}||{param_name}[{i}]||"
                    title = f"||{param_name}||"

                ax.plot(
                    steps,
                    data_array[:, i],
                    label=label,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    alpha=alpha,
                    color=color,
                )

        ax.set_title(title)

    def _plot_statistics_curves(
        self, ax: plt.Axes, steps: List[int], stats_array: np.ndarray, field_name: str, shape: Any
    ) -> None:
        """Plot statistics curves."""
        if isinstance(shape, tuple) and len(shape) == 2 and isinstance(shape[1], str):
            k, list_type = shape
            for i in range(k):
                if list_type == "list_1d":
                    ax.plot(steps, stats_array[:, i], label=f"||{field_name}[{i}]||", linewidth=0.2)
                elif list_type == "list_2d":
                    ax.plot(steps, stats_array[:, i], label=f"||{field_name}[{i}]||_F", linewidth=0.2)
                else:
                    ax.plot(steps, stats_array[:, i], label=f"||{field_name}[{i}]||", linewidth=0.2)
            ax.set_title(f"Stats: {field_name}")
        elif len(shape) == 0:
            # Scalar statistics
            ax.plot(steps, stats_array, label=f"{field_name}", linewidth=0.2)
            ax.set_title(f"Stats: {field_name}")
        elif len(shape) == 1:
            k = shape[0]
            for i in range(k):
                ax.plot(steps, stats_array[:, i], label=f"{field_name}[{i}]", linewidth=0.2)
            ax.set_title(f"Stats: {field_name}")
        elif len(shape) == 2:
            k, n = shape
            for i in range(k):
                ax.plot(steps, stats_array[:, i], label=f"||{field_name}[{i}]||", linewidth=0.2)
            ax.set_title(f"Stats: ||{field_name}||")
        else:
            for i in range(shape[0]):
                ax.plot(steps, stats_array[:, i], label=f"||{field_name}[{i}]||", linewidth=0.2)
            ax.set_title(f"Stats: ||{field_name}||")

    def _configure_axis(
        self, ax: plt.Axes, param_data: Dict[str, Any], plot_data: Dict[str, Any], plot_mode: str
    ) -> None:
        """Configure axis properties and legend."""
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        # Add legend if not too many lines
        available_shapes = []
        if plot_mode in ["both", "regular"] and "regular" in param_data:
            available_shapes.append(plot_data[param_data["regular"][0]])
        if plot_mode in ["both", "polyak"] and "polyak" in param_data:
            available_shapes.append(plot_data[param_data["polyak"][0]])

        if available_shapes:
            shape = available_shapes[0]
            legend_threshold = 4 if plot_mode == "both" else 8
            if isinstance(shape, tuple) and len(shape) == 2 and isinstance(shape[1], str):
                if shape[0] <= legend_threshold:
                    ax.legend(fontsize=7)
            elif len(shape) == 0:
                # Always show legend for scalar parameters
                ax.legend(fontsize=7)
            elif len(shape) >= 1 and shape[0] <= legend_threshold:
                ax.legend(fontsize=7)

    def _finalize_plot(self, fig: plt.Figure, axes: List[plt.Axes], plot_idx: int, plot_mode: str, stats: bool) -> None:
        """Finalize the plot with title and layout."""
        # Remove empty subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        # Set title based on plot mode and stats inclusion
        if stats:
            title_map = {
                "both": "Parameter and Statistics Convergence",
                "regular": "Regular Parameter and Statistics Convergence",
                "polyak": "Polyak Averaged Parameter and Statistics Convergence",
            }
        else:
            title_map = {
                "both": "Parameter Convergence (Regular + Polyak)",
                "regular": "Regular Parameter Convergence",
                "polyak": "Polyak Averaged Parameter Convergence",
            }

        plt.suptitle(title_map[plot_mode], fontsize=16, fontweight="bold")
        plt.tight_layout()


def create_em_logger(file_path: Union[str, Path]) -> AsyncHDF5Logger:
    """Create an AsyncHDF5Logger."""
    return AsyncHDF5Logger(file_path)


def load_em_log(file_path: Union[str, Path]) -> HDF5LogReader:
    """Load an EM training log for analysis."""
    return HDF5LogReader(file_path)
