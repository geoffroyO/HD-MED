from pathlib import Path

import grain.python as grain
import numpy as np


class NumpyDataSource(grain.RandomAccessDataSource):
    def __init__(self, file_path: str, split: str = "train", test_split: float = 0.2, seed: int = 42):
        self.split = split
        self.test_split = test_split
        self.seed = seed

        file_path: Path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file = np.load(file_path)
        signals = np.array(file["signals"])
        self._create_splits(signals)

    def _create_splits(self, signals: np.ndarray):
        np.random.seed(self.seed)
        indices = np.arange(len(signals))
        np.random.shuffle(indices)
        split_index = int(len(signals) * (1 - self.test_split))
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        if self.split == "train":
            self.signals = signals[train_indices]
        else:
            self.signals = signals[test_indices]

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.signals[index]