"""Provides compatible for some tools from the bioembeddings package."""

from typing import Final, List, Optional, Sequence, Iterable, TypeVar
from .core import IntEncoder, get_default_int_encoder
from torch import Tensor

from transformers import AutoModel, AutoTokenizer
import torch

T = TypeVar("T")


def chunks(inp: Sequence[T], n: int) -> Iterable[Sequence[T]]:
    """Yield successive n-sized chunks from inp.

    From stack overflow:
    https://stackoverflow.com/questions/312443/
    how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(inp), n):
        yield inp[i:(i + n)]


class ESMCEncoderWrapper:
    """Featurizer acting on integer encoded sequences.

    Note that this wrapping is not optimized, and may be slow for each encoding
    invocation.

    """

    ESMC_huggingface_name: Final = "biohub/ESMC-600M-hf"

    def __init__(
        self,
        integer_encoder: Optional[IntEncoder],
        device: str,
        flatten: bool = True,
        per_protein: bool = True,
        batch_size: int = 32,
    ) -> None:
        """Store options.

        Arguments:
        ---------
        integer_encoder:
            IntEncoder instance whose batch_decode method will be used to create
            string-represented sequences from integer encoded sequences. Must correspond
            to how sequences later passed to batch_encode were originally
            integer-encoded.
        device:
            torch device identifier (e.g., "cuda")
        flatten:
            If per_protein is True, ignored. Else, if true, we return a matrix
            of shape (b, f) for batched input. Otherwise, the returned values
            are organized by sequence, shape (b,
            n_res, 1152)
        per_protein:
            If True, we average over all residues in each sample to produce a single
            1152-sized vector. Setting this to False keeps embeddings separate per
            amino acid. False is more expressive but may have a massive memory
            footprint.
        batch_size:
            Number of examples to process at once. Bigger values use more memory
            and probably don't help performance.

        """
        if integer_encoder is None:
            integer_encoder = get_default_int_encoder()
        self.device = device
        self.flatten = flatten
        self.tokenizer= AutoTokenizer.from_pretrained(self.ESMC_huggingface_name)

        # Load the model
        self.esmc = AutoModel.from_pretrained(self.ESMC_huggingface_name).to(device)
        self.integer_encoder = integer_encoder
        self.per_protein = per_protein
        self.batch_size = batch_size

    def vectorized_encode(self, int_encoded: Tensor) -> Tensor:
        """Encode iterable of sequences.

        Arguments:
        ---------
        int_encoded:
            Tensor of shape (b, d): b specifies the sample (i.e., batch dimension). Each
            element is an integer encoded (not one-hot) encoded element. Should
            be encoded using the integer_encoder created an initialization.

        Returns:
        -------
        Returns Tensor with first dimension the batch dimension.

        """
        str_features = self.integer_encoder.batch_decode(int_encoded)
        inputs = self.tokenizer(str_features, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # generate embeddings
        with torch.no_grad():
            embedding_repr = self.esmc(**inputs)

        emb: Tensor = embedding_repr.last_hidden_state

        if self.per_protein:
            # average over residues, not batch examples
            return emb[:,1:-1,:].mean(dim=1)
        elif self.flatten:
            return emb[:,1:-1,:].flatten(start_dim=1)
        else:
            return emb[:,1:-1,:]

    def batch_encode(self, int_encoded: Tensor) -> Tensor:
        """Encode iterable of sequences.

        Unlike vectorized_encode, input is processed in chunks.

        Arguments:
        ---------
        int_encoded:
            Tensor of shape (b, d): b specifies the sample (i.e., batch dimension). Each
            element is an integer encoded (not one-hot) encoded element. Should
            be encoded using the integer_encoder created an initialization.

        Returns:
        -------
        Returns Tensor with first dimension the batch dimension.

        """
        processed: List[Tensor] = []
        for piece in chunks(int_encoded, self.batch_size):
            processed.append(self.vectorized_encode(piece))
        return torch.concatenate(processed, axis=0)  # type: ignore[arg-type]
