"""Contains descriptions of all considered datasets.

Objects here allow data files to be associated with experiment IDs
and other data attributes. Routines are provided to conveniently
query available files.

updated by LW: allow user to create new Dataspec from their own data

"""

from typing import Final, Union
from dataclasses import dataclass, asdict
from itertools import chain
from pathlib import Path

## ------------------------  Notes  -------------------------------(1)
#1. Fixed project-level references / constants.
#2. Final = type hint meaning "之后不能在assign其他的value了".
#3. 自己define的名字: Final = 比如path(),比如是一个sequence.

# this file only contains structure for residues 5-187 inclusive
# this is not yet correctly taken into account in some places.
LEGACY_SARSCOV2_FILENAME: Final = Path("RBD_amaro.pdb")

# this has all residues
SARSCOV2_FILENAME: Final = Path("RBD_7DX5.pdb")

SARS_COV2_SEQ: Final = (
    "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKL"
    "NDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSK"
    "VGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTN"
    "GVGYQPYRVVVLSFELLHAPATVCGPKKST"
)


## ------------------------  Notes  -------------------------------(2)
#1. @dataclass 就是Python auto-generates的class不用写 __init__, etc.
#2. 写了一个class叫 Datasepc
#3. order=True: sortable/comparable
#4. frozen=True: attributes cannot be changed after creation


@dataclass(order=True, frozen=True)
class DataSpec:
    """Describes all files associated with a given experiment."""

    name: str  # convienient name labeling the experiment.
    train_filename: Path  # local path of train csv
    valid_filename: Path  # local path of valid csv
    test_filename: Path  # local path of valid csv
    index: int  # positive unique labeling experiment.



## ------------------------  Updated Notes  -------------------------------(3)
def create_dataspec(
    name: str,  # convienient name labeling the experiment.
    train_filename: Path,  # local path of train csv
    valid_filename: Path,  # local path of valid csv
    test_filename: Path,  # local path of valid csv
    index: int =0,  # positive unique labeling experiment.
    ) -> DataSpec:
    
    """Create a DataSpec for a user-provided dataset."""
    return DataSpec(
        name=name,
        train_filename=train_filename,
        valid_filename=valid_filename,
        test_filename=test_filename,
        index=index,
    )



## ------------------------  Notes  -------------------------------(3)
#1. List of DataSpec objects; each element describes one experiment.
#2. Example: CORE_DATA_SPECS[0].name -> "base"
#3. attribute = object 里存的数据 ; method = object 能执行的函数,就是def的！

CORE_DATA_SPECS: Final = [
    DataSpec(
        name="base",
        train_filename=Path("train_data.csv"),
        valid_filename=Path("valid_data.csv"),
        test_filename=Path("test_data.csv"),
        index=0,
    ),
    DataSpec(
        name="B1351",
        train_filename=Path("not_norm_train_data_B1351.csv"),
        valid_filename=Path("not_norm_valid_data_B1351.csv"),
        test_filename=Path("not_norm_test_data_B1351.csv"),
        index=1,
    ),
    DataSpec(
        name="E484K",
        train_filename=Path("not_norm_train_data_E484K.csv"),
        valid_filename=Path("not_norm_valid_data_E484K.csv"),
        test_filename=Path("not_norm_test_data_E484K.csv"),
        index=2,
    ),
    DataSpec(
        name="N501Y",
        train_filename=Path("not_norm_train_data_N501Y.csv"),
        valid_filename=Path("not_norm_valid_data_N501Y.csv"),
        test_filename=Path("not_norm_test_data_N501Y.csv"),
        index=3,
    ),
    DataSpec(
        name="BA1",
        train_filename=Path("not_norm_train_data_omicron_BA1.csv"),
        valid_filename=Path("not_norm_valid_data_omicron_BA1.csv"),
        test_filename=Path("not_norm_test_data_omicron_BA1.csv"),
        index=4,
    ),
    DataSpec(
        name="BA2",
        train_filename=Path("not_norm_train_data_omicron_BA2.csv"),
        valid_filename=Path("not_norm_valid_data_omicron_BA2.csv"),
        test_filename=Path("not_norm_test_data_omicron_BA2.csv"),
        index=5,
    ),
    DataSpec(
        name="wuhan_omicron",
        train_filename=Path("not_norm_train_data_omicron_Wuhan_Hu_1.csv"),
        valid_filename=Path("not_norm_valid_data_omicron_Wuhan_Hu_1.csv"),
        test_filename=Path("not_norm_test_data_omicron_Wuhan_Hu_1.csv"),
        index=6,
    ),
    DataSpec(
        name="wuhan",
        train_filename=Path("not_norm_train_data_Wuhan_Hu_1.csv"),
        valid_filename=Path("not_norm_valid_data_Wuhan_Hu_1.csv"),
        test_filename=Path("not_norm_test_data_Wuhan_Hu_1.csv"),
        index=7,
    ),
]

ALT_DATA_SPECS: Final = [
    DataSpec(
        name="base_trainval-rng42",
        train_filename=Path("train_rng42.csv"),
        valid_filename=Path("val_rng42.csv"),
        test_filename=Path("test_data.csv"),
        index=8,
    ),
    DataSpec(
        name="base_trainval-rng596",
        train_filename=Path("train_rng596.csv"),
        valid_filename=Path("val_rng596.csv"),
        test_filename=Path("test_data.csv"),
        index=9,
    ),
    DataSpec(
        name="base_trainvaltest-rng789",
        train_filename=Path("train_fullresplit_rng789.csv"),
        valid_filename=Path("val_fullresplit_rng789.csv"),
        test_filename=Path("test_fullresplit_rng789.csv"),
        index=10,
    ),
    DataSpec(
        name="base_dedup",
        train_filename=Path("base_dedup_train.csv"),
        valid_filename=Path("base_dedup_valid.csv"),
        test_filename=Path("base_dedup_test.csv"),
        index=11,
    ),
    DataSpec(
        name="base_mutsplit_1",
        train_filename=Path("base_mut1_train.csv"),
        valid_filename=Path("base_mut1_valid.csv"),
        test_filename=Path("test_data.csv"),
        index=12,
    ),
    DataSpec(
        name="base_mutsplit_12",
        train_filename=Path("base_mut12_train.csv"),
        valid_filename=Path("base_mut12_valid.csv"),
        test_filename=Path("test_data.csv"),
        index=13,
    ),
    DataSpec(
        name="base_mutsplit_123",
        train_filename=Path("base_mut123_train.csv"),
        valid_filename=Path("base_mut123_valid.csv"),
        test_filename=Path("test_data.csv"),
        index=14,
    ),
    DataSpec(
        name="base_mutsplit_1234",
        train_filename=Path("base_mut1234_train.csv"),
        valid_filename=Path("base_mut1234_valid.csv"),
        test_filename=Path("test_data.csv"),
        index=15,
    ),
    DataSpec(
        name="base_mutsplit_12345",
        train_filename=Path("base_mut12345_train.csv"),
        valid_filename=Path("base_mut12345_valid.csv"),
        test_filename=Path("test_data.csv"),
        index=16,
    ),
    DataSpec(
        name="small_base",
        train_filename=Path("small_base_train.csv"),
        valid_filename=Path("small_base_valid.csv"),
        test_filename=Path("test_data.csv"),
        index=17,
    ),
]

DATA_SPECS: Final = CORE_DATA_SPECS + ALT_DATA_SPECS

MAX_DATASPEC_INDEX: Final = max(x.index for x in DATA_SPECS)


## ------------------------  Notes  -------------------------------(4)
#1. define一个新的function: def resolve_dataspec(identifier)
#2. 而这个identifier可以是str, int, DataSpec 其中一个
#3. -> 是type annotation, 有或没有都可以！

def resolve_dataspec(identifier: Union[str, int, DataSpec]) -> DataSpec:
    """Return data specification matching name or index.

    Arguments:
    ---------
    identifier:
        Either a string, integer, or Dataspec. If a Dataspec, directly
        returned. Otherwise, matched against all possible index and
        name fields of stored experiments.

    Returns:
    -------
    Dataspec

    """
    if isinstance(identifier, DataSpec):
        return identifier
    for x in DATA_SPECS:
        if identifier == x.name or identifier == x.index:
            return x
    else:
        raise ValueError("No matching DataSpec found.")


## ------------------------  Notes  -------------------------------(5)
#1. Define function to check everything in DATA_SEPCS are unique

def _sanity_check() -> None:
    """Perform basic checks to avoid hard-to-find typographical errors."""
    # check to make sure no fields are duplicated anywhere in the core specs
    _lists = [list(asdict(x).values()) for x in CORE_DATA_SPECS]
    _all = list(chain.from_iterable(_lists))
    assert len(_all) == len(set(_all))
    del _lists
    del _all

    # check to make sure there are no duplicate names in all specs
    names = [x.name for x in DATA_SPECS]
    assert len(names) == len(set(names))
    del names

    # check to make sure there are no duplicate indices in all specs
    indices = [x.index for x in DATA_SPECS]
    assert len(indices) == len(set(indices))
    del indices

    # check to make sure that all index ints are >= 0
    assert all(x.index >= 0 for x in DATA_SPECS)


_sanity_check()
