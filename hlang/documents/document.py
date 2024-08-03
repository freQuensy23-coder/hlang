from typing import Any
from abc import ABC, abstractmethod


class ABCDocument(ABC):
    @abstractmethod
    def __init__(self, content: Any):
        pass

    @abstractmethod  # TODO think about name
    def embed_text(self) -> str:
        pass
