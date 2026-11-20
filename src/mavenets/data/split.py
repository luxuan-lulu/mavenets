"""Utilities for splitting datasets."""

from pathlib import Path
from typing import Iterable, Optional, Tuple, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from ..util import compute_mutation_distances


def _save_splits(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: Path,
    name: str,
) -> None:
    """Save splits in CSV format."""

    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame.to_csv(
        output_dir / f"{name}_train.csv",
        index=False
    )
    valid_frame.to_csv(
        output_dir / f"{name}_valid.csv",
        index=False
    )
    test_frame.to_csv(
        output_dir / f"{name}_test.csv",
        index=False
    )


def random_split(
    frame: pd.DataFrame,
    train_size: float = 0.8,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    shuffle: bool = True,
    stratify_col: Optional[str] = None,
    save: bool = False,
    output_dir: Optional[Path] = None,
    name: str = "dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Randomly split a DataFrame into train, validation, and test sets."""

    stratify = frame[stratify_col] if stratify_col is not None else None
    
    train_frame, tmp_frame = train_test_split(
        frame,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    valid_fraction = valid_size / (valid_size + test_size)

    if stratify_col is not None:
        tmp_stratify = tmp_frame[stratify_col]
    else:
        tmp_stratify = None

    valid_frame, test_frame = train_test_split(
        tmp_frame,
        train_size=valid_fraction,
        random_state=random_state,
        shuffle=shuffle,
        stratify=tmp_stratify,
    )

    if save:
        if output_dir is None:
            raise ValueError(
                "output_dir is required when save=True."
            )

        _save_splits(
            train_frame,
            valid_frame,
            test_frame,
            output_dir,
            name
        )

    return train_frame, valid_frame, test_frame


def split_by_group(
    frame: pd.DataFrame,
    test_group: Iterable[Any],
    train_group: Optional[Iterable[Any]] = None,
    valid_group: Optional[Iterable[Any]] = None,
    split_col: Optional[str] = None,
    valid_size: float = 0.1,
    random_state: int = 42,
    stratify_col: Optional[str] = None,
    shuffle: bool = True,
    sequence_col: Optional[str] = None,
    reference_sequence: Optional[str] = None,
    save: bool = False,
    output_dir: Optional[Path] = None,
    name: str = "dataset"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """Split a DataFrame by groups.

    Three split modes are supported:

    1. Hold-out test mode:
       Specify test_group only. The selected group(s) are used as the
       test set, and all remaining rows are randomly split into training
       and validation sets.

    2. Hold-out test mode with selected training groups:
       Specify test_group and train_group, but not valid_group.
       The selected test group(s) are held out completely. Rows belonging
       to train_group form the development pool, which is then randomly
       split into training and validation sets.

    3. Explicit group mode:
       Specify train_group, valid_group, and test_group. Each dataset is
       constructed directly from the specified groups.

    If split_col is not provided, mutation distances relative to
    reference_sequence are calculated and used as the grouping variable.

    stratify_col is used only when a random train/validation split is
    performed (Modes 1 and 2).
    """

    # Use an existing column for splitting if provided.
    # Otherwise calculate mutation counts from the reference sequence.

    if split_col is not None:
        if split_col not in frame.columns:
            raise ValueError(f"'{split_col}' was not found in the DataFrame.")

    else:
        if reference_sequence is None or sequence_col is None:
            raise ValueError("reference_sequence and sequence_col are required when split_col is not provided.")

        if sequence_col not in frame.columns:
            raise ValueError(f"'{sequence_col}' was not found in the DataFrame.")

        frame = frame.copy()
        split_col = "mut_num"
        frame[split_col] = compute_mutation_distances(frame[sequence_col].tolist(), reference_sequence)


    if test_group is None:
        raise ValueError("test_group is required.")

    if train_group is None and valid_group is not None:
        raise ValueError("valid_group cannot be provided without train_group.")

    test_values = set(test_group)

    ### Mode1: test_group only ####
    if train_group is None:
        test_frame = frame[frame[split_col].isin(test_values)].copy()
        dev_frame = frame[~frame[split_col].isin(test_values)].copy()

        stratify = dev_frame[stratify_col] if stratify_col is not None else None

        train_frame, valid_frame = train_test_split(
        dev_frame,
        test_size=valid_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
        )

    ### Mode2: selected test_group + train_group ####
    elif valid_group is None:
        train_values = set(train_group)
        if train_values & test_values:
            raise ValueError("train and test split value overlap.")
        test_frame = frame[frame[split_col].isin(test_values)].copy()
        dev_frame = frame[frame[split_col].isin(train_values)].copy()

        stratify = dev_frame[stratify_col] if stratify_col is not None else None

        train_frame, valid_frame = train_test_split(
        dev_frame,
        test_size=valid_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
        )
      
    ### Mode3: explicit train/val/test groups ####
    else:
        train_values = set(train_group)
        valid_values = set(valid_group)

        if train_values & valid_values:
            raise ValueError("train and validation split value overlap.")
        if train_values & test_values:
            raise ValueError("train and test split value overlap.")
        if valid_values & test_values:
            raise ValueError("validation and test split value overlap.")

        train_frame = frame[frame[split_col].isin(train_values)].copy()
        valid_frame = frame[frame[split_col].isin(valid_values)].copy()
        test_frame = frame[frame[split_col].isin(test_values)].copy()

    if save:
        if output_dir is None:
            raise ValueError("output_dir is required when save=True.")

        _save_splits(
            train_frame,
            valid_frame,
            test_frame,
            output_dir,
            name
        )

    return train_frame, valid_frame, test_frame
