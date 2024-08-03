from openlangchain.documents.document import ABCDocument


class SimpleDocument(ABCDocument):
    def __init__(self, content: str):
        self.content = content

    def embed_text(self) -> str:
        return self.content
