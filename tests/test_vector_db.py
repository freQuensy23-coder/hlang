from unittest import TestCase

import torch

from hlang.documents.simple_document import Document
from hlang.vectordb.vector_db import VectorStorage


class TestVectorDatabase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embeddings = torch.rand(1_000, 312)
        cls.documents = [Document(f"Document {i}") for i in range(5_000)]
        cls.db = VectorStorage(cls.documents, cls.embeddings)

    def test_search(self):
        query = torch.rand(1, 312)
        results = self.db.search(query)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(isinstance(doc, Document) for doc in results))
        self.assertTrue(all(doc in self.documents for doc in results))

    def test_search_correct_response(self):
        query = self.embeddings[-1]
        results = self.db.search(query)
        self.assertEqual(
            results[0], self.documents[-1]
        )
