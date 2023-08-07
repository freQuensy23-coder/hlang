from typing import Type

import torch
from sentence_transformers import SentenceTransformer

from hlang.documents.document import ABCDocument


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def get_embedding_dimension(self):
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, texts: [str]) -> torch.Tensor:
        return self.model.encode(texts, convert_to_tensor=True).cpu()

    def embed_documents(self, documents: [Type[ABCDocument]]) -> torch.Tensor:
        return self.model.encode([document.embed_text() for document in documents], convert_to_tensor=True).cpu()
