from unittest import TestCase

from openlangchain.documents.key_value_document import KeyValueDocument
from openlangchain.documents.simple_document import SimpleDocument


class TestDocument(TestCase):

    def test_simple_document(self):
        doc = SimpleDocument(content="Hello, world!")
        self.assertEqual(doc.embed_text(), "Hello, world!")

    def test_key_value_document(self):
        doc = KeyValueDocument(key="description of something", content="Name of something")
        self.assertEqual(
            doc.embed_text(),
            "description of something"
        )
