"""Utilities for splitting datasets."""

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def _select_columns(
    frame: pd.DataFrame,
    id_col: str,
    sequence_col: str,
    signal_col: str,
) -> pd.DataFrame:
    """Keep only columns needed downstream."""
    return frame[[id_col, sequence_col, signal_col]].copy()

def _save_splits(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: Path,
    name: str,
) -> None:
    """Save splits in MAVENets CSV format."""

    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame.to_csv(
        output_dir / f"{name}_train.csv",
        index=False,
        header=False,
    )
    valid_frame.to_csv(
        output_dir / f"{name}_valid.csv",
        index=False,
        header=False,
    )
    test_frame.to_csv(
        output_dir / f"{name}_test.csv",
        index=False,
        header=False,
    )


def random_split(
    frame: pd.DataFrame,
    id_col: str,
    sequence_col: str,
    signal_col: str,
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

    train_frame = _select_columns(
        train_frame, id_col, sequence_col, signal_col
    )
    valid_frame = _select_columns(
        valid_frame, id_col, sequence_col, signal_col
    )
    test_frame = _select_columns(
        test_frame, id_col, sequence_col, signal_col
    )


    if save:
        if output_dir is None or id_col is None or sequence_col is None or signal_col is None:
            raise ValueError(
                "output_dir, id_col, sequence_col, and signal_col are required "
                "when save=True."
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
    mutation_count_col: str,
    train_num: Iterable[int],
    valid_num: Iterable[int],
    test_num: Iterable[int],
    id_col: str,
    sequence_col: str,
    signal_col: str,
    save: bool = False,
    output_dir: Optional[Path] = None,
    name: str = "dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame by mutation number."""

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

    train_frame = _select_columns(
        train_frame, id_col, sequence_col, signal_col
    )
    valid_frame = _select_columns(
        valid_frame, id_col, sequence_col, signal_col
    )
    test_frame = _select_columns(
        test_frame, id_col, sequence_col, signal_col
    )


    if save:
        if output_dir is None or id_col is None or sequence_col is None or signal_col is None:
            raise ValueError(
                "output_dir, id_col, sequence_col, and signal_col are required "
                "when save=True."
            )

        _save_splits(
            train_frame,
            valid_frame,
            test_frame,
            output_dir,
            name
        )

    return train_frame, valid_frame, test_frame
