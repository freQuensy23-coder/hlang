from typing import Type

import chromadb
import torch

from hlang.documents.document import ABCDocument


class VectorStorage:
    def __init__(self, documents: [Type[ABCDocument]], embeddings: torch.Tensor):
        self.documents = documents
        self.embeddings = embeddings
        self._client = chromadb.Client()
        self._collection = self._client.create_collection(f"documents_{id(self)}")
        self._collection.add(
            ids=[str(i) for i in range(len(documents))],
            embeddings=[e.tolist() for e in embeddings],
        )

    def search(self, v: torch.Tensor, top_k: int = 5) -> [Type[ABCDocument]]:
        results = self._collection.query(
            query_embeddings=v.numpy().reshape(1, -1),
            n_results=top_k
        )
        ids = results['ids'][0]
        return [self.documents[int(i)] for i in ids]
