"""Tools for loading data."""

from typing import (
    Final,
    Tuple,
    Iterable,
    Literal,
    Union,
    Dict,
    overload,
    Optional,
    Protocol,
    TypeVar,
    Generic,
    List
)
import pandas as pd  # type: ignore
from torch.utils.data import TensorDataset
from torch import Tensor, tensor, float32, int32
from torch.utils.data import Dataset
from pathlib import Path
from torch_geometric.data import Data  # type: ignore
from .spec import CORE_DATA_SPECS, DataSpec, resolve_dataspec, SARSCOV2_FILENAME
from .featurize import (
    get_alphabet,
    get_default_int_encoder,
    int_to_floatonehot,
    IntEncoder,
    SKT_protocol,
    Whiten,
    NullTransform,
)
from .graph import get_graph
from .split import random_split, split_by_mutation_num

# column names for labeling loaded MAVE experiment csvs.
CSV_RID_CNAME: Final = "seq_id"
SEQ_CNAME: Final = "sequence"
SIGNAL_CNAME: Final = "signal"
EXPERIMENT_CNAME: Final = "experiment_index"


# Type variable for the item type returned by a dataset's __getitem__
_T_co = TypeVar("_T_co", covariant=True)


class SizedDataset(Protocol[_T_co]):
    """Protocol for a Dataset that implements __len__ and __getitem__."""

    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    def __getitem__(self, idx: int, /) -> _T_co:
        """Return the item at the given index."""
        ...


## -------------------------------------  Notes ----------------------------------------()
#1. SequenceDataset 是一个 wrapper: 它把已经存在的 PyTorch Dataset 包起来，同时额外保存 raw amino-acid sequences。
#2. It stores two main things:
#   _dataset
#       -> 原来的 dataset
#       -> e.g. TensorDataset(features, signals, exp_ids)
#
#   _sequences
#       -> 对应每个 sample 的 raw protein sequence
#       -> e.g. ("ACDEF...", "GHIKL...", ...)
#3. Usage: 
# len(seq_dataset) -> sample 数量
# seq_dataset[i] -> 原 dataset 的第 i 个 sample -> (feature, signal, exp_id)
# seq_dataset.get_sequence(i) -> 第 i 条 raw sequence
# seq_dataset.sequences -> 所有 raw sequences
# seq_dataset.dataset -> 原来的 underlying dataset

class SequenceDataset(Dataset[_T_co], Generic[_T_co]):
    """Wrapper dataset that adds sequence and metadata access to an underlying dataset.

    This class wraps an existing Dataset (such as TensorDataset or DNSEDataset)
    and provides access to the raw amino acid sequences and optional metadata corresponding to each
    data point. When used as a standard Dataset (via indexing or iteration),
    it behaves identically to the wrapped dataset.

    The generic type parameter _T_co represents the item type returned by
    __getitem__, which is preserved from the underlying dataset.

    Example:
    -------
    ```
    base_dataset = TensorDataset(features, signals, exp_ids)
    sequences = ("ACDEF...", "GHIKL...", ...)
    metadata = {"mut_num": (1, 2)}
    seq_dataset = SequenceDataset(base_dataset, sequences, metadata)

    # Standard dataset access returns same as base_dataset
    item = seq_dataset[0]  # Returns (features[0], signals[0], exp_ids[0])

    # Sequence access
    seq = seq_dataset.get_sequence(0)  # Returns "ACDEF..."
    all_seqs = seq_dataset.sequences  # Returns tuple of all sequences

    sample_metadata = seq_dataset.get_metadata(0)
    all_metadata = seq_dataset.metadata
    ```

    """

    _dataset: SizedDataset[_T_co]
    _sequences: Tuple[str, ...]
    _metadata: Dict[str, Tuple[object, ...]]

    def __init__(
        self, dataset: SizedDataset[_T_co], sequences: Tuple[str, ...], metadata: Optional[Dict[str, Tuple[object, ...]]] = None
    ) -> None:
        """Initialize with a base dataset, corresponding sequences, and optional metadata.

        Arguments:
        ---------
        dataset:
            The underlying Dataset to wrap. All standard Dataset operations
            are delegated to this object. Must implement __len__.
        sequences:
            Tuple of raw amino acid sequences, one per data point.
            Must have the same length as the dataset.
        metadata:
            Optional dictionary containing additional metadata for each data point.
            Keys are metadata field names, and values are tuples of the same length
            as the dataset. If not provided, no metadata are stored.
        Raises:
        ------
        ValueError:
            If the number of sequences or metadata values doesn't match the dataset length.

        """
        super().__init__()
        if len(sequences) != len(dataset):
            raise ValueError(
                f"Number of sequences ({len(sequences)}) must match "
                f"dataset length ({len(dataset)})"
            )
    
        if metadata is not None:
            for key, values in metadata.items():
                if len(values) != len(dataset):
                    raise ValueError(
                        f"Metadata column '{key}' has length {len(values)}, "
                        f"but dataset length is {len(dataset)}."
                    )

        
        self._dataset = dataset
        self._sequences = sequences
        self._metadata = {} if metadata is None else dict(metadata)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> _T_co:
        """Return the item at the given index from the underlying dataset."""
        return self._dataset[idx]

    def get_sequence(self, idx: int) -> str:
        """Return the raw sequence at the given index.

        Arguments:
        ---------
        idx:
            Index of the sequence to retrieve.

        Returns:
        -------
        The raw amino acid sequence string at the given index.

        """
        return self._sequences[idx]

    def get_metadata(self, idx: int) -> Dict[str, object]:
        """Return the metadata for the given index.

        Arguments:
        ---------
        idx:
            Index of the data point to retrieve metadata.

        Returns:
        -------
        Dictionary containing metadata for the given index.

        """
        return {key: values[idx] for key, values in self._metadata.items()}

    @property
    def sequences(self) -> Tuple[str, ...]:
        """Return all raw sequences.

        Returns:
        -------
        Tuple of all raw amino acid sequences in the dataset.

        """
        return self._sequences

    @property
    def metadata(self) -> Dict[str, Tuple[object, ...]]:
        """Return metadata columns stored alongside the dataset."""
        return self._metadata

    @property
    def dataset(self) -> SizedDataset[_T_co]:
        """Return the underlying wrapped dataset.

        Returns:
        -------
        The underlying Dataset instance.

        """
        return self._dataset


class DNSEDataset(Dataset):
    """Graph dataset with identical edges but varying labels.

    This creates a dataset with Dynamic Node embeddings and Static Edges. At
    initialization, connectivity and edge features are given which apply to
    all served examples. Other tensors are also provided; these tensors
    are indexed along their first axis to create examples.

    Example:
    -------
    ```
    edge_ind, edge_feat = get_graph(...) # get SARS-COV2 structural graph
    node_feats = torch.randn(5,201) # generate face node labels for 5 structures
    d = DNSEDataset(edge_attr=edge_feat,edge_index=edge_ind,x=node_feats)
    ```

    `d` is now a ataset of 5 examples, each with the same graph but different node
    features. Note feature information is placed under the attribute name `x`. The
    name of the attribute is defined by the name of the argument used; multiple
    tensors may be specified as multiple arguments.

    """

    def __init__(
        self,
        /,
        edge_attr: Tensor,
        edge_index: Tensor,
        **kwargs,
    ) -> None:
        """Store data.

        Arguments are keyword-only.

        Arguments:
        ---------
        edge_attr:
            Tensor of shape (n_edges, n_edge_Feats) coining the features of each edge.
        edge_index:
            Tensor of shape (n_edges, 2) containing the start and end of each edge.
        **kwargs:
            Tensors which are indexed along their leading axis when serving examples.

        """
        super().__init__()
        self.edge_feat = edge_attr
        self.edge_index = edge_index
        if len(kwargs) == 0:
            raise ValueError("Must provide at least one field tensor.")
        self.tensor_dict: Dict[str, Tensor] = kwargs

    def __len__(self) -> int:
        """Return number of data pairs."""
        first_item = next(iter(self.tensor_dict.values()))
        return first_item.shape[0]

    def __getitem__(self, idx: int) -> Data:
        """Slice each underlying tensor, assemble Data object, serve."""
        additional_fields = {}
        for key, value in self.tensor_dict.items():
            # Views
            additional_fields[key] = value[idx]
        # note that batch_based indexing may be affected by the names
        # of attributes
        data = Data(
            **additional_fields,
            edge_attr=self.edge_feat,
            edge_index=self.edge_index,
        )

        return data


## -------------------------------------  Notes ----------------------------------------()
#1. will be used in the next function named: _get_aggregate_mave_csv
#2. read one headerless MAVE CSV and label its three columns as seq_id, sequence, and signal.

# applies column names solely based on column order.
def _mave_csv_read(f: Path, has_header: bool = False) -> pd.DataFrame:
    """Read csv files of a given format from disk and label columns.

    Format is assumed to be:

    If has_header is False, the file is assumed to contain three columns:
    identification number, sequence, and signal.

    If has_header is True, column names are read directly from the file and
    all columns are preserved.     

    """
    if has_header:
        frame = pd.read_csv(f)

    else:
        frame = pd.read_csv(f, header=None)
        frame.columns = [CSV_RID_CNAME, SEQ_CNAME, SIGNAL_CNAME]
    return frame


## -------------------------------------  Notes ----------------------------------------()
#1. resolve_dataspec 在spec.py define, convert int, str, Datasepc to Dataspec
#2. 得到Dataspec, 从中的identifier提取文件名称，利用上面定义的_mave_csv_read 读取csv file, 转成三列column，然后外加一个experiment index defined in Dataspec 然后把多个csv concat成一个df

def _get_aggregate_mave_csv(
    specs: Union[Iterable[int], Iterable[str], Iterable[DataSpec]],
    identifier: str,
    directory: Path,
) -> pd.DataFrame:
    """Read selection of csv files based on a given identifier.

    Arguments:
    ---------
    specs:
        Iterable of identifiers to pass to resolve_dataset.
    identifier:
        Used to access each data spec via getattr. Likely "train_filename",
        "valid_filename", or "test_filename".
    directory:
        Where to look for the specified files obtained via the getattr call.

    Returns:
    -------
    DataFrame containing loaded sequences and the corresponding integer
    experiment identifiers under EXPERIMENT_CNAME.

    """
    frames = []

    # load datasets, record index
    for sp in (resolve_dataspec(x) for x in specs):
        tf = _mave_csv_read(directory / getattr(sp, identifier))
        tf[EXPERIMENT_CNAME] = sp.index
        frames.append(tf)

    return pd.concat(frames)


## -------------------------------------  Notes ----------------------------------------()

def _standardize_columns(
    frame: pd.DataFrame,
    id_col: str,
    sequence_col: str,
    signal_col: str,
) -> pd.DataFrame:
    return frame.rename(
        columns={
            id_col: CSV_RID_CNAME,
            sequence_col: SEQ_CNAME,
            signal_col: SIGNAL_CNAME,
        }
    )


## -------------------------------------  Notes ----------------------------------------()

def _get_data_frames(
    *,
    # raw data path
    frame: Optional[pd.DataFrame] = None,
    frame_path: Optional[Path] = None,
    id_col: Optional[str] = None,
    sequence_col: Optional[str] = None,
    signal_col: Optional[str] = None,
    split_type: Optional[str] = None,
    split_kwargs: Optional[dict] = None,
    # already-split CSV path
    train_path: Optional[Path] = None,
    valid_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    split_has_header: bool = False,
)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get train, validation, and test DataFrames."""

    # Check which input mode is being used
    has_raw_input = frame is not None or frame_path is not None
    has_split_input = (
        train_path is not None
        or valid_path is not None
        or test_path is not None
    )

    # Raw data and pre-split data cannot be provided together
    if has_raw_input and has_split_input:
        raise ValueError(
            "Provide either raw data or already-split CSV files, not both."
        )

    # 1. Raw CSV -> DataFrame
    if frame is not None and frame_path is not None:
        raise ValueError("Provide either frame or frame_path, not both.")

    if frame_path is not None:
        frame = pd.read_csv(frame_path)

    # 2. Raw DataFrame -> split.py
    if frame is not None:

        if id_col is None or sequence_col is None or signal_col is None:
            raise ValueError(
                "id_col, sequence_col, and signal_col are required "
                "when loading raw data."
            )

        frame = _standardize_columns(
            frame,
            id_col,
            sequence_col,
            signal_col,
        )

        split_kwargs = dict(split_kwargs or {})

        if split_type == "random":
            return random_split(frame, **split_kwargs)

        if split_type == "mutation_num":
            split_kwargs["sequence_col"] = SEQ_CNAME
            return split_by_mutation_num(frame, **split_kwargs)

        raise ValueError(
                "split_type must be 'random' or 'mutation_num'."
            )

    # 3. Already-split CSV files
    if train_path is not None and valid_path is not None and test_path is not None:

        train_frame = _mave_csv_read(
            train_path,
            has_header=split_has_header,
        )
        valid_frame = _mave_csv_read(
            valid_path,
            has_header=split_has_header,
        )
        test_frame = _mave_csv_read(
            test_path,
            has_header=split_has_header,
        )

        # If files have headers, standardize their required column names.
        if split_has_header:
            if id_col is None or sequence_col is None or signal_col is None:
                raise ValueError(
                    "id_col, sequence_col, and signal_col are required "
                    "for headered split CSV files."
                    )

            train_frame = _standardize_columns(
                train_frame,
                id_col,
                sequence_col,
                signal_col,
            )
            valid_frame = _standardize_columns(
                valid_frame,
                id_col,
                sequence_col,
                signal_col,
            )
            test_frame = _standardize_columns(
                test_frame,
                id_col,
                sequence_col,
                signal_col,
            )

        return train_frame, valid_frame, test_frame

    raise ValueError(
        "Provide raw data (frame/frame_path) or "
        "train_path, valid_path, and test_path."
    )


## -------------------------------------  Notes ----------------------------------------()
#1.convert df to embedding and ready for model to learn
#2.Use IntEncoder.batch_encode() defined in core.py to convert all sequences into integer tensor
#3.depedning on feat_type to select embedding type (e.g. integer, onehot, t5) Can improve!!!
#4.Convert signal column to float32 tensor -> prediction target y.
#5.Convert experiment_index to int32 tensor -> dataset/experiment ID.
                                                                                            
def _process_table(
    frame: pd.DataFrame, feat_type: str, int_encoder: IntEncoder, device: str
) -> Tuple[Tensor, Tensor, Tensor, Tuple[str, ...]]:
    """Transform data frame into processed tensors.

    Arguments:
    ---------
    frame:
        Data frame to process. Should have columns corresponding to SEQ_CNAME,
        SIGNA_CNAME, and EXPERIMENT_CNAME variables. Likely from _mave_csv_read or
        _get_aggregate_mave_csv.
    feat_type:
        How to featurize data. "onehot" corresponds to one-hot features
        (float32), "integer" corresponds to integer encoding, "t5" uses
        encodings from a pretrained transformer.
    int_encoder:
        IntEncoder instance to perform integer encoding.
    device:
        torch device specifier. Only used for t5 inference.

    Returns:
    -------
    Four values: First is the featurized sequences, second is the signal to
    fit against, third contains dataset ids, fourth is a tuple of raw sequences.

    """
    raw_sequences: Tuple[str, ...] = tuple(frame.loc[:, SEQ_CNAME].tolist())
    int_encoded = int_encoder.batch_encode(frame.loc[:, SEQ_CNAME])
    if feat_type == "onehot":
        encoded = int_to_floatonehot(int_encoded, num_classes=len(int_encoder.alphabet))
    elif feat_type == "integer":
        encoded = int_encoded
    elif feat_type == "t5":
        from .featurize.t5 import T5EncoderWrapper

        enc = T5EncoderWrapper(
            integer_encoder=int_encoder, device=device, per_protein=True
        )
        encoded = enc.batch_encode(int_encoded).cpu()
    elif feat_type == "esmc":
        from .featurize.esmc import ESMCEncoderWrapper

        enc = ESMCEncoderWrapper(
            integer_encoder=int_encoder, device=device, per_protein=True
        )
        encoded = enc.batch_encode(int_encoded).cpu()
    else:
        raise ValueError("Unknown featurization type: {}".format(feat_type))

    signal = tensor(frame.loc[:, SIGNAL_CNAME].to_numpy(), dtype=float32)
    dset_id = tensor(frame.loc[:, EXPERIMENT_CNAME].to_numpy(), dtype=int32)
    return encoded, signal, dset_id, raw_sequences


## -------------------------------------  Notes ----------------------------------------()
# 1. @overload 只是类型说明，不是真正实现。
# 2. 作用：
# 1) include_test=False  -> 返回 (train_dataset, valid_dataset)
# 2) include_test=True   -> 返回 (train_dataset, valid_dataset, test_dataset)
# 3) 不传 include_test   -> 默认按 False 处理，返回两个 dataset


@overload
def get_datasets(
    *,
    device: str,
    train_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    val_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    test_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    train_frame: Optional[pd.DataFrame] = None,
    valid_frame: Optional[pd.DataFrame] = None,
    test_frame: Optional[pd.DataFrame] = None,
    metadata_cols: Optional[List[str]] = ...,
    feat_type: Literal["integer", "onehot", "t5", "esmc"] = ...,
    graph: bool = ...,
    graph_sequence_window_size: int = ...,
    graph_n_distance_feats: int = ...,
    graph_distance_cutoff: float = ...,
    parent_path: Path = ...,
    include_test: Literal[False],
    whiten: Optional[bool] = ...,
) -> Tuple[SequenceDataset, SequenceDataset]:
    ...


@overload
def get_datasets(
    *,
    device: str,
    train_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    val_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    test_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    train_frame: Optional[pd.DataFrame] = None,
    valid_frame: Optional[pd.DataFrame] = None,
    test_frame: Optional[pd.DataFrame] = None,
    metadata_cols: Optional[List[str]] = ...,
    feat_type: Literal["integer", "onehot", "t5", "esmc"] = ...,
    graph: bool = ...,
    graph_sequence_window_size: int = ...,
    graph_n_distance_feats: int = ...,
    graph_distance_cutoff: float = ...,
    parent_path: Path = ...,
    include_test: Literal[True],
    whiten: Optional[bool] = ...,
) -> Tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    ...


@overload
def get_datasets(
    *,
    device: str,
    train_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    val_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    test_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = ...,
    train_frame: Optional[pd.DataFrame] = None,
    valid_frame: Optional[pd.DataFrame] = None,
    test_frame: Optional[pd.DataFrame] = None,
    metadata_cols: Optional[List[str]] = ...,
    feat_type: Literal["integer", "onehot", "t5", "esmc"] = ...,
    graph: bool = ...,
    graph_sequence_window_size: int = ...,
    graph_n_distance_feats: int = ...,
    graph_distance_cutoff: float = ...,
    parent_path: Path = ...,
    include_test: Literal[False] = ...,
    whiten: Optional[bool] = ...,
) -> Tuple[SequenceDataset, SequenceDataset]:
    ...


## -------------------------------------  Notes ----------------------------------------()
#1. get_datasets() 把 raw CSV data 读进来、做 sequence embedding, preprocessing，然后包装成可以直接用于 PyTorch training 的 Dataset
#2. * 的意思是后面的所有参数都必须用（参数名=值）的方式传递
#3. train/val/test specs: 用DataSpec, int and str来指定到底使用哪些experiments, iterable 意思是遍历所有提供的experiment, 默认等于None也就是CORE_DATA_SPECS
#4. feat_type 其实是embedding,如果后续improve embedding, 则需要找到这里
#5. whiten 就是 Z score的standardize，define在tranform.py里面，只有在T5的时候调用
#6. 真正的code截止到all frames, 就是load train/val/test(if include)的csv and convert to concated df if include lots of dataset once
#7. _process_table()把df转成integer representation and feat type defined embedding
#8. whiten/NullTranfrom 决定data的pre-processing
#9. 根据GNN or not 来创建base dataset：graph=False 就是 TensorDataset(encoded, signal, experiment_id)
#graph=True:就是DNSEDataset(...)
#10. 再用 SequenceDataset 包起来: SequenceDataset(base_datset,raw_sequence), 这样既可以用pytorch dataset也可以访问protein sequence
#11. 最后的输出就是 SequenceDataset wrapper for train/valid/test_dataset


def get_datasets(  # noqa: C901
    *,
    device: str,
    train_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = None,
    val_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = None,
    test_specs: Union[None, Iterable[DataSpec], Iterable[int], Iterable[str]] = None,
    train_frame: Optional[pd.DataFrame] = None,
    valid_frame: Optional[pd.DataFrame] = None,
    test_frame: Optional[pd.DataFrame] = None,
    metadata_cols: Optional[List[str]] = None,
    feat_type: Literal["integer", "onehot", "t5", "esmc"] = "integer",
    graph: bool = False,
    graph_sequence_window_size: int = 10,
    graph_n_distance_feats: int = 10,
    graph_distance_cutoff: float = 2.5,
    parent_path: Path = Path(),
    include_test: bool = False,
    whiten: Optional[bool] = None,
) -> Union[
    Tuple[SequenceDataset, SequenceDataset],
    Tuple[SequenceDataset, SequenceDataset, SequenceDataset],
]:
    """Load, featurize, and return SARSCOV2 data for training and evaluation.

    Loads target signal and sequences from disk, and if graph is specified reads
    a file describing the 3d structure of the protein. If graph is True, the
    underlying datasets are DNSEDataset instances; else, TensorDatasets are used.
    All returned datasets are wrapped in SequenceDataset, which provides access
    to the raw amino acid sequences via the get_sequence() method and sequences
    property.

    If include_test is True, 3 datasets are returned: train, validation, and test.
    If False, only train and validation are returned.

    Arguments:
    ---------
    device:
        Torch device on which to place the entire datasets. Likely "cuda" or "cpu".
    train_specs:
        None or Iterable of ints or strings used to specify what datasets to place in
        the training dataset. If integers, compared against the index of the specs; if
        a string, compared against the names. If None, all data sets are used.
    val_specs:
        None or Iterable of ints or strings used to specify what datasets to place in
        the validation dataset. If integers, compared against the index of the specs;
        if a string, compared against the names. If None, all data sets are used.
    test_specs:
        None or Iterable of ints or strings used to specify what datasets to place in
        the test dataset. If integers, compared against the index of the specs;
        if a string, compared against the names. If None, all data sets are used. This
        information is only used if include_test is True.
    feat_type:
        Featurization used; only "integer", "onehot", and "t5" are accepted. "integer"
        corresponds to a vector with one integer entry per amino acid determining
        the residue type. "onehot" creates a 0-1 vector that is longer with the same
        information (see torch.nn.functional.one_hot). Note that the one hot is
        converted to the float32 dtype. "t5" uses embeddings from a pretrained
        T5 model from hugging face. Note that t5 may trigger the download
        of the model which is approximately 10GB.
    graph:
        If True, returned datasets are DNSEDataset instances based on a
        structure/sequence graph. Edges are directed and featurized; see graph_* and
        ..graph.get_graph
    graph_sequence_window_size:
        When graph is True, passed to ..graph.get_graph. Defines how close two
        residues must be in primary sequence to gain a sequence-based
        connection.
    graph_n_distance_feats:
        When graph is True, passed to ..graph.get_graph.  Distance in sequence
        and 3D space are expanded using a set of radial basis functions. This
        argument controls the number of basis functions.
    graph_distance_cutoff:
        When graph is True, passed to ..graph.get_graph. When two residues are
        within this distance (in Angstroms), they are connected via a distance-based
        contact.
    parent_path:
        Path object specifying where to look for csv (and if specified, structure)
        files.
    include_test:
        If True, we return 3 datasets: train, validation, and test. If false,
        only train and validation are returned.
    whiten:
        Whether to whiten features. If True, a whitening transform is trained on
        the statistics of the training set and used to transform the training,
        validation, and test sets. If None, we use whitening on t5 transformed
        data but not elsewhere.

    Returns:
    -------
    (2 or 3)-Tuple of SequenceDataset instances (train, val, [test]) wrapping the
    underlying datasets. When iterated, they return (feat, signal, dataset_index)
    for TensorDataset-based or pyg Data objects for graph-based datasets. The
    SequenceDataset wrapper provides additional sequence access via get_sequence()
    and the sequences property. See include_test for whether 2 or 3 datasets are
    returned.

    """
    if feat_type not in ("integer", "onehot", "t5", "esmc"):
        raise ValueError("Only integer, onehot, esmc or t5 featurization is supported.")

    if whiten is None:
        whiten = feat_type in ("t5","esmc")

    if whiten:
        post_transform: SKT_protocol = Whiten()
    else:
        post_transform = NullTransform(copy=False)

# Update by LW: choose DataFrame route OR original DataSpec route
    frames_provided = (
        train_frame is not None
        or valid_frame is not None
        or test_frame is not None
    )

    if not frames_provided:

        # Original DataSpec workflow
        if train_specs is None:
            train_specs = CORE_DATA_SPECS

        if val_specs is None:
            val_specs = CORE_DATA_SPECS

        if test_specs is None:
            test_specs = CORE_DATA_SPECS

        train_frame = _get_aggregate_mave_csv(
            specs=train_specs, identifier="train_filename", directory=parent_path
        )

        valid_frame = _get_aggregate_mave_csv(
            specs=val_specs, identifier="valid_filename", directory=parent_path
        )

        if include_test:
            test_frame = _get_aggregate_mave_csv(
                specs=test_specs, identifier="test_filename", directory=parent_path
            )

    # From here onward, both routes are identical
    assert train_frame is not None
    assert valid_frame is not None

    all_frames = [train_frame, valid_frame]

    if include_test:
        assert test_frame is not None
        all_frames.append(test_frame)

    # Ensure experiment column exists for all frames
    for frame in all_frames:
        if EXPERIMENT_CNAME not in frame.columns:
            frame[EXPERIMENT_CNAME] = 0

    # Preserve only user-selected metadata
    train_metadata = None
    valid_metadata = None
    test_metadata = None

    if metadata_cols is not None:
        train_metadata = {
            col: tuple(train_frame[col])
            for col in metadata_cols
        }

        valid_metadata = {
            col: tuple(valid_frame[col])
            for col in metadata_cols
        }

        if include_test:
            test_metadata = {
                col: tuple(test_frame[col])
                for col in metadata_cols
            }

    enc = get_default_int_encoder()

    # make sure that there are no amino acids in the data not in our standard
    # alphabet. We use a standard alphabet to maintain featurization stability
    # across possibly smaller input datasets.
    alpha = get_alphabet(pd.concat(all_frames), SEQ_CNAME)
    if not set(alpha).issubset(set(enc.alphabet)):
        raise ValueError("Data contains residues not represented fixed alphabet.")

    train_encoded, train_signal, train_dset_id, train_sequences = _process_table(
        train_frame,
        feat_type=feat_type,
        int_encoder=enc,
        device=device,
    )

    post_transform.fit(train_encoded)
    train_encoded = post_transform.transform(train_encoded)

    valid_encoded, valid_signal, valid_dset_id, valid_sequences = _process_table(
        valid_frame,
        feat_type=feat_type,
        int_encoder=enc,
        device=device,
    )

    valid_encoded = post_transform.transform(valid_encoded)

    # Initialize test variables - will be set if include_test is True
    test_base: Optional[SizedDataset[object]] = None
    test_sequences: Optional[Tuple[str, ...]] = None

    if include_test:
        test_encoded, test_signal, test_dset_id, test_sequences = _process_table(
            test_frame,
            feat_type=feat_type,
            int_encoder=enc,
            device=device,
        )
        test_encoded = post_transform.transform(test_encoded)

    if graph:
        edge_labels, edge_features = get_graph(
            structure=str(parent_path / SARSCOV2_FILENAME),
            max_cutoff=graph_distance_cutoff,
            min_cutoff=0.0,
            num_distance_features=graph_n_distance_feats,
            window_size=graph_sequence_window_size,
            node_offset=0,
        )

        train_base: Dataset = DNSEDataset(
            edge_attr=edge_features.to(device),
            edge_index=edge_labels.to(device),
            x=train_encoded.to(device),
            y=train_signal.to(device),
            experiment=train_dset_id.to(device),
        )
        valid_base: Dataset = DNSEDataset(
            edge_attr=edge_features.to(device),
            edge_index=edge_labels.to(device),
            x=valid_encoded.to(device),
            y=valid_signal.to(device),
            experiment=valid_dset_id.to(device),
        )

        if include_test:
            test_base = DNSEDataset(
                edge_attr=edge_features.to(device),
                edge_index=edge_labels.to(device),
                x=test_encoded.to(device),
                y=test_signal.to(device),
                experiment=test_dset_id.to(device),
            )
    else:
        train_base = TensorDataset(
            train_encoded.to(device), train_signal.to(device), train_dset_id.to(device)
        )
        valid_base = TensorDataset(
            valid_encoded.to(device), valid_signal.to(device), valid_dset_id.to(device)
        )
        if include_test:
            test_base = TensorDataset(
                test_encoded.to(device), test_signal.to(device), test_dset_id.to(device)
            )

    # Wrap datasets with SequenceDataset to provide sequence access
    train_dataset: SequenceDataset = SequenceDataset(train_base, train_sequences, metadata=train_metadata)
    valid_dataset: SequenceDataset = SequenceDataset(valid_base, valid_sequences, metadata=valid_metadata )

    if include_test and test_base is not None and test_sequences is not None:
        test_dataset: SequenceDataset = SequenceDataset(test_base, test_sequences, metadata=test_metadata)
        return train_dataset, valid_dataset, test_dataset

    return train_dataset, valid_dataset
