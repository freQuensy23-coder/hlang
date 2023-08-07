from abc import abstractmethod, ABC
from typing import Type

import torch

from hlang.documents.document import ABCDocument


class ABCEmbedder(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def embed_text(self, texts: [str]) -> torch.Tensor:
        pass

    @abstractmethod
    def embed_document(self, document: Type[ABCDocument]) -> torch.Tensor:
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        pass
