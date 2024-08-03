from typing import Type

import faiss
import torch

from openlangchain.documents.document import ABCDocument


class VectorDatabase:
    def __init__(self, documents: [Type[ABCDocument]], embeddings: torch.Tensor):
        self.documents = documents
        self.embeddings = embeddings
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(len(embeddings[0])))
        self.index.add(embeddings)

    def search(self, v: torch.Tensor, top_k: int = 5) -> [Type[ABCDocument]]:
        v, indexes = self.index.search(v, top_k)
        return [self.documents[i] for i in indexes[0]]
