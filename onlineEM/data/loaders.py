import grain.python as grain

from chex import dataclass


@dataclass
class DataLoaderOutput:
    """Output from create_dataloader function."""

    train_loader: grain.DataLoader
    test_loader: grain.DataLoader
    batch_size: int
    num_train_steps: int
    num_val_steps: int
    num_epochs: int


def _create_dataloader(
    source: grain.RandomAccessDataSource,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    num_epochs: int = 1,
) -> grain.DataLoader:
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=num_epochs,
        shuffle=shuffle,
        seed=seed,
    )

    batch_transform = grain.Batch(batch_size=batch_size, drop_remainder=True)
    transforms = [batch_transform]

    # Create PyGrain DataLoader
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=transforms,
        worker_count=0,
    )


def create_dataloader(
    data_source: grain.RandomAccessDataSource,
    file_path: str,
    test_split: float = 0.2,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    num_epochs: int = 1,
) -> DataLoaderOutput:
    source_train = data_source(file_path, split="train", test_split=test_split, seed=seed)
    source_test = data_source(file_path, split="test", test_split=test_split, seed=seed)

    train_loader = _create_dataloader(source_train, batch_size, shuffle, seed, num_epochs)
    test_loader = _create_dataloader(source_test, batch_size, shuffle, seed, num_epochs)

    num_train_steps = len(source_train) // batch_size - 1
    num_val_steps = len(source_test) // batch_size - 1

    return DataLoaderOutput(
        train_loader=train_loader,
        test_loader=test_loader,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        num_val_steps=num_val_steps,
        num_epochs=num_epochs,
    )
