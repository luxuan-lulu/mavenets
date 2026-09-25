"""Routines to featurize sequence data."""

from typing import (
    List,
    Dict,
    overload,
    Literal,
    Union,
    Iterable,
    Optional,
    Protocol,
    Final,
    cast,
)
from functools import lru_cache
import pandas as pd  # type: ignore
import torch
from torch.nn.functional import one_hot
from torch import float32, int64
from ..spec import SARS_COV2_SEQ

# known amino acid codes. Defining them statically here allows
# reproduciblity if models are training on datasets lacking chemical coverage.
BASE_ALPHA: Final = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "X",
]

## -------------------------------------  Notes ----------------------------------------()
#1. frame.loc[:, column_name] 把特定的一整个column全部拿出来, 然后return 所有的不重复的作为sorted list

def get_alphabet(frame: pd.DataFrame, column_name: str) -> List[str]:
    """Return all possible letters."""
    column_values = cast(List[str], list(frame.loc[:, column_name]))  # type: ignore[arg-type]
    return sorted(set("".join(column_values)))

## -------------------------------------  Notes ----------------------------------------()
#1. Create amino-acid <-> integer lookup dictionaries. (enumerate就是给他assign一个index)
#2. encoder: residue -> index  {"A" : 0,...}
#3. decoder: index -> residue  {"0" : A,...}
#4. Sanity check that the encoder and decoder are exact inverses using SARS_COV2_SEQ

def encoder_dict(alphabet: List[str]) -> Dict[str, int]:
    """Create encoder dictionary.

    Arguments:
    ---------
    alphabet:
        list of letters in dictionary. Assumed have unique elements.

    Returns:
    -------
    Dictionary mapping alphabet members to integers.

    """
    return {letter: ind for ind, letter in enumerate(alphabet)}


def decoder_dict(alphabet: List[str]) -> Dict[int, str]:
    """Create encoder dictionary.

    Arguments:
    ---------
    alphabet:
        list of letters in dictionary. Assumed have unique elements.

    Returns:
    -------
    Dictionary mapping integers to alphabet members.

    """
    return dict(enumerate(alphabet))


def _dict_sanity_check() -> None:
    letters = list(SARS_COV2_SEQ)
    enc = encoder_dict(BASE_ALPHA)
    dec = decoder_dict(BASE_ALPHA)
    for let in letters:
        encoded = enc[let]
        if dec[encoded] != let:
            raise ValueError(
                "Encoder does not satisfy round-trip equivalence in sanity check."
            )


_dict_sanity_check()


class _p_encode(Protocol):
    @overload
    def __call__(
        self, target: str, tensor: Literal[True], device: Optional[str] = ...
    ) -> torch.Tensor:
        ...

    @overload
    def __call__(
        self, target: str, tensor: Literal[False], device: Optional[str] = ...
    ) -> List[int]:
        ...

    def __call__(
        self, target: str, tensor: bool, device: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        ...


class _p_decode(Protocol):
    @overload
    def __call__(
        self, target: Union[Iterable[int], torch.Tensor], cast: Literal[True] = ...
    ) -> str:
        ...

    @overload
    def __call__(self, target: Iterable[int], cast: Literal[False] = ...) -> str:
        ...

    def __call__(
        self, target: Union[Iterable[int], torch.Tensor], cast: bool = True
    ) -> str:
        ...


## -------------------------------------  Notes ----------------------------------------()
#1.把 amino-acid sequence 在string 和 integer representation 之间转换
#2.根据之前定义的BASE_ALPHA 以及encode_map/decode_map来建立两个loop up table
#3._encode/_decode就可以根据table来转化一个sequence
#4.batch_encode/batch_decode可以把所有的sequence stack在一起
#5.并且可以查看alphabet的length

class IntEncoder:
    """Encodes strings as integer lists or tensors."""

    def __init__(
        self, alphabet: List[str], lru_cache_size: Optional[int] = None
    ) -> None:
        """Create backing dictionaries.

        Arguments:
        ---------
        alphabet:
            list of letters in dictionary. Must have unique elements.
        lru_cache_size:
            If a positive integer, then we wrap encoding in an LRU cache of this size.
            Note that when serving mutable encodings (e.g., tensors), this option
            can share memory/objects if two equal but distinct arguments are encoded.
            When using batch encoding this likely will not happen since there is an
            additional stacking step.

        """
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("Alphabet does not comprise unique elements.")
        self.alphabet: Final = alphabet
        self.encode_map: Final = encoder_dict(alphabet)
        self.decode_map: Final = decoder_dict(alphabet)
        if lru_cache_size:
            # it seems like lru_cache is picking up on only the first overloaded type
            # definition, so we override the type.
            self.encode: _p_encode = lru_cache(maxsize=lru_cache_size)(self._encode)  # type: ignore
            self.decode: _p_decode = lru_cache(maxsize=lru_cache_size)(self._decode)  # type: ignore
        else:
            self.encode = self._encode
            self.decode = self._decode

    @overload
    def _encode(
        self, target: str, tensor: Literal[True], device: Optional[str] = ...
    ) -> torch.Tensor:
        ...

    @overload
    def _encode(
        self, target: str, tensor: Literal[False], device: Optional[str] = ...
    ) -> List[int]:
        ...

    def _encode(
        self, target: str, tensor: bool, device: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        """Encode single example.

        Arguments:
        ---------
        target:
            Single string to encode.
        tensor:
            If true, encoded content is returned as a Tensor. Else, returned as a
            list of ints.
        device:
            Device to make tensor on. Likely "cpu" or "cuda".

        Returns:
        -------
        List of integers or a Tensor.

        """
        encoded = [self.encode_map[x] for x in target]
        if tensor:
            return torch.tensor(encoded, dtype=torch.int32, device=device)
        else:
            return encoded

    @overload
    def _decode(
        self, target: Union[Iterable[int], torch.Tensor], cast: Literal[True] = ...
    ) -> str:
        ...

    @overload
    def _decode(self, target: Iterable[int], cast: Literal[False] = ...) -> str:
        ...

    def _decode(self, target: Union[Iterable[int], torch.Tensor], cast: bool = True) -> str:
        """Decode single example.

        Translates from integers to a string.

        Arguments:
        ---------
        target:
           Iterable of integers or a torch.Tensor to transform into a string.
        cast:
            If True, we call int() on each symbol before attempting lookup.
            Note that when passing a Tensor, cast must be True, as it is used to look
            up a dictionary keyed by ints.

        Returns:
        -------
        string

        """
        if cast:
            return "".join([self.decode_map[int(x)] for x in target])
        else:
            # cast=False overload guarantees target is Iterable[int]
            return "".join([self.decode_map[x] for x in target])  # type: ignore[index]

    def batch_decode(
        self, target: Union[Iterable[Iterable[int]], torch.Tensor]
    ) -> List[str]:
        """Decode iterable of examples.

        Translates from integers to strings.

        Arguments:
        ---------
        target:
           Iterable of integer iterables or a 2D Tensor to transform into strings.

        Returns:
        -------
        List of strings.

        """
        return [self.decode(x, cast=True) for x in target]

    def batch_encode(
        self, targets: Iterable[str], device: Optional[str] = None
    ) -> torch.Tensor:
        """Encode multiple examples into single tensor.

        Only supports encoding into a Tensor.
        """
        encoded = [self.encode(x, tensor=True, device=device) for x in targets]
        return torch.stack(encoded, dim=0)

    def __len__(self) -> int:
        """Return number of known symbols."""
        return len(self.encode_map)

## -------------------------------------  Notes ----------------------------------------()
# 1. get_default_int_encoder():
#    用固定的 BASE_ALPHA 创建一个默认的 IntEncoder object。
#    如果指定 cache_size，则同时设置 LRU cache；默认不开 cache
# 2. 检查整个 IntEncoder 的 encode -> decode 流程是否正确。

def get_default_int_encoder(cache_size: Optional[int] = None) -> IntEncoder:
    """Return default integer encoder."""
    return IntEncoder(BASE_ALPHA, lru_cache_size=cache_size)


def _default_encoder_sanity_check() -> None:
    enc = get_default_int_encoder()
    sample = "".join(enc.alphabet)
    if sample != enc.decode(enc.encode(sample, tensor=False)):
        raise ValueError(
            "Default encoder does not satisfy round-trip equivalence in sanity check."
        )


_default_encoder_sanity_check()


## -------------------------------------  Notes ----------------------------------------()
# 1. 用onehot form pytorch来从刚才的integer encoding转换成one-hot encoding
# 2. 先把 int32 转成 int64，因为 PyTorch one_hot() 要求 class index 是 int64/LongTensor。
# 3. one-hot 结果最后再转成 float32，方便送进 neural network。
# 4. shape: (N, L) -> (N, L, num_classes)

def int_to_floatonehot(int_form: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    """Transform 32-bit integer encoding to a float32 one-hot encoding.

    Arguments:
    ---------
    int_form:
        Tensor to Transform. Cast to int64 internally.
    num_classes:
        Number of classes (size) to use in one hot encoding. If -1, the number of
        classes present in the data is used. See one_hot.

    Returns:
    -------
    float32 one hot encoding.

    """
    int_encoding: torch.Tensor = one_hot(
        int_form.to(int64),
        num_classes=num_classes,
    )
    encoded: torch.Tensor = int_encoding.to(float32)  # type: ignore[no-any-return]
    return encoded
