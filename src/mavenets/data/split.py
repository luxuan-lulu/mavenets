"""Utilities for splitting datasets."""

from pathlib import Path
from typing import Iterable, Optional, Tuple

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


def split_by_mutation_num(
    frame: pd.DataFrame,
    train_num: Iterable[int],
    valid_num: Iterable[int],
    test_num: Iterable[int],
    sequence_col: Optional[str] = None,
    mutation_count_col: Optional[str] = None,
    reference_sequence: Optional[str] = None,
    save: bool = False,
    output_dir: Optional[Path] = None,
    name: str = "dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame by mutation number."""


    # Use existing mutation counts if available.
    # Otherwise calculate them from the reference sequence.

    if mutation_count_col is not None:
        if mutation_count_col not in frame.columns:
            raise ValueError(
                f"'{mutation_count_col}' was not found in the DataFrame."
            )

    else:
        if reference_sequence is None or sequence_col is None:
            raise ValueError(
                "reference_sequence and sequence_col are required when mutation_count_col is not provided."
            )

        frame = frame.copy()
        mutation_count_col = "mut_num"
        frame[mutation_count_col] = compute_mutation_distances(
            frame[sequence_col].tolist(),
            reference_sequence,
        )

    train_counts = set(train_num)
    valid_counts = set(valid_num)
    test_counts = set(test_num)

    if train_counts & valid_counts:
        raise ValueError("train and validation mutation numbers overlap.")

    if train_counts & test_counts:
        raise ValueError("train and test mutation numbers overlap.")

    if valid_counts & test_counts:
        raise ValueError("validation and test mutation numbers overlap.")

    train_frame = frame[
        frame[mutation_count_col].isin(train_counts)
    ].copy()

    valid_frame = frame[
        frame[mutation_count_col].isin(valid_counts)
    ].copy()

    test_frame = frame[
        frame[mutation_count_col].isin(test_counts)
    ].copy()

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
