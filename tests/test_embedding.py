from unittest import TestCase

import torch

from hlang.documents.simple_document import Document
from hlang.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder


class TestSentenceEmbedding(TestCase):
    def setUp(self):
        self.model = SentenceTransformerEmbedder(
            model_name="sergeyzh/rubert-tiny-turbo"
        )

    def test_text_embedding(self):
        text = "Hello, world!"
        embedding = self.model.embed_text([text])
        self.assertEqual(embedding.shape, (1, 312))

    def test_text_embedding_batch(self):
        text = ["Hello, world!", "How are you?"]
        embedding = self.model.embed_text(text)
        self.assertEqual(embedding.shape, (2, 312))

    def test_text_embedding_empty(self):
        text = ""
        embedding = self.model.embed_text([text])
        self.assertEqual(embedding.shape, (1, 312))

    def test_embed_document(self):
        text = "Hello, world!"
        doc = Document(content=text)
        embedding = self.model.embed_documents([doc])
        self.assertEqual(embedding.shape, (1, 312))
        self.assertLess(
            torch.norm(embedding - self.model.embed_text([text])).item(),
            1e-5
        )

    def test_embed_document_batch(self):
        text = ["Hello, world!", "How are you?"]
        docs = [Document(content=t) for t in text]
        embedding = self.model.embed_documents(docs)
        self.assertEqual(embedding.shape, (2, 312))
        self.assertLess(
            torch.norm(embedding - self.model.embed_text(text)).item(),
            1e-5
        )

